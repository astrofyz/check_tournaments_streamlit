#!/usr/bin/env python3
"""Alternate overlap checker: uses GET /players/{id}/tournaments instead of per-tournament results.

Compare with tourn_check_web.py (results+teamMembers on every seed ∪ intersection tournament).
Same report shape for summaries; input includes overlap_strategy for benchmarks.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import requests

DEFAULT_BASE = "https://api.rating.chgk.net"
DEFAULT_TIMEOUT = 60
MAX_ITEMS_PER_PAGE = 512


class RatingAPIError(Exception):
    """HTTP or unexpected payload from api.rating.chgk.net."""


def editor_surnames_from_tournament(row: dict[str, Any]) -> list[str]:
    editors = row.get("editors")
    if not isinstance(editors, list):
        return []
    out: list[str] = []
    for ed in editors:
        if isinstance(ed, dict):
            s = ed.get("surname")
            if s is not None and str(s).strip():
                out.append(str(s).strip())
    return out


def parse_text_lines(raw: str) -> list[str]:
    lines: list[str] = []
    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def parse_player_ids_from_text(text: str) -> list[int]:
    ids: list[int] = []
    for line in parse_text_lines(text):
        try:
            ids.append(int(line))
        except ValueError as e:
            raise ValueError(f"Invalid player id (expected integer): {line!r}") from e
    if not ids:
        raise ValueError("No player ids in input")
    return ids


def parse_tournament_lines_from_text(text: str) -> list[str]:
    lines = parse_text_lines(text)
    if not lines:
        raise ValueError("No tournament lines in input")
    return lines


class RatingClient:
    def __init__(
        self,
        base_url: str,
        timeout: float = DEFAULT_TIMEOUT,
        verbose: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.verbose = verbose
        self.session = requests.Session()
        self.session.headers["Accept"] = "application/json"

    def _url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return f"{self.base_url}{path}"

    def get(self, path: str, params: list[tuple[str, str | int]] | dict[str, Any] | None = None) -> Any:
        url = self._url(path)
        if self.verbose:
            print(f"GET {url} params={params}", file=sys.stderr)
        r = self.session.get(url, params=params, timeout=self.timeout)
        if r.status_code == 429:
            retry_after = r.headers.get("Retry-After")
            wait = float(retry_after) if retry_after and retry_after.isdigit() else 2.0
            time.sleep(wait)
            r = self.session.get(url, params=params, timeout=self.timeout)
        if not r.ok:
            raise RatingAPIError(f"HTTP {r.status_code} for {r.url}\n{r.text[:500]}")
        return r.json()

    def fetch_tournament_item(self, tournament_id: int) -> dict[str, Any] | None:
        """GET /tournaments/{id}; None if 404 or unexpected body."""
        url = self._url(f"/tournaments/{tournament_id}")
        if self.verbose:
            print(f"GET {url}", file=sys.stderr)
        r = self.session.get(url, timeout=self.timeout)
        if r.status_code == 429:
            retry_after = r.headers.get("Retry-After")
            wait = float(retry_after) if retry_after and retry_after.isdigit() else 2.0
            time.sleep(wait)
            r = self.session.get(url, timeout=self.timeout)
        if r.status_code == 404:
            return None
        if not r.ok:
            raise RatingAPIError(f"HTTP {r.status_code} for {r.url}\n{r.text[:500]}")
        data = r.json()
        return data if isinstance(data, dict) else None

    def fetch_tournaments_by_name(
        self,
        name_substring: str,
        date_end_strictly_after: str | None = None,
        items_per_page: int = MAX_ITEMS_PER_PAGE,
    ) -> list[dict[str, Any]]:
        all_rows: list[dict[str, Any]] = []
        page = 1
        while True:
            params: list[tuple[str, str | int]] = [
                ("name[]", name_substring),
                ("page", page),
                ("itemsPerPage", items_per_page),
            ]
            if date_end_strictly_after is not None:
                params.insert(1, ("dateEnd[strictly_after]", date_end_strictly_after))
            chunk = self.get("/tournaments", params=params)
            if not isinstance(chunk, list):
                raise RatingAPIError(f"Unexpected /tournaments response type: {type(chunk)}")
            all_rows.extend(chunk)
            if len(chunk) < items_per_page:
                break
            page += 1
        return all_rows

    def fetch_intersections(self, tournament_id: int) -> list[dict[str, Any]]:
        data = self.get(f"/tournaments/{tournament_id}/intersections")
        if not isinstance(data, list):
            raise RatingAPIError(f"Unexpected intersections response type: {type(data)}")
        return data

    def fetch_player_tournaments(self, player_id: int) -> list[dict[str, Any]]:
        """GET /players/{id}/tournaments/ — full list in one response (no pagination)."""
        data = self.get(f"/players/{player_id}/tournaments/")
        if not isinstance(data, list):
            raise RatingAPIError(
                f"Unexpected /players/{{id}}/tournaments/ response type: {type(data)}"
            )
        return data


_thread_local = threading.local()


def _thread_local_client(base_url: str, timeout: float, verbose: bool) -> RatingClient:
    """One Session per thread (requests sessions are not thread-safe)."""
    key = (base_url, timeout, verbose)
    if getattr(_thread_local, "_client_key", None) != key:
        _thread_local._client_key = key
        _thread_local._client = RatingClient(base_url, timeout=timeout, verbose=verbose)
    return _thread_local._client


def intersection_ids_from_response(rows: list[dict[str, Any]]) -> list[int]:
    out: set[int] = set()
    for row in rows:
        tid = row.get("id")
        if tid is not None:
            out.add(int(tid))
    return sorted(out)


def tournament_id_from_player_row(row: dict[str, Any]) -> int | None:
    for key in ("idtournament", "idTournament"):
        v = row.get(key)
        if v is not None:
            return int(v)
    return None


def strip_leading_until_alnum(line: str) -> str:
    """Drop leading characters until the first letter or number (Unicode-aware)."""
    for i, ch in enumerate(line):
        if ch.isalnum():
            return line[i:]
    return ""


def tournament_line_is_seed_id(line: str) -> bool:
    """True if the whole line is decimal digits only (tournament id)."""
    return bool(line) and line.isdigit()


def tournaments_matching_all_words(
    client: RatingClient,
    words: list[str],
    date_end_after: str | None,
) -> list[dict[str, Any]]:
    """Each word is a separate name[] API query; keep tournaments whose id appears in every result set."""
    sets: list[set[int]] = []
    rows_by_id: dict[int, dict[str, Any]] = {}
    for w in words:
        chunk = client.fetch_tournaments_by_name(w, date_end_after)
        ids: set[int] = set()
        for r in chunk:
            tid = r.get("id")
            if tid is None:
                continue
            tid_i = int(tid)
            ids.add(tid_i)
            if tid_i not in rows_by_id:
                rows_by_id[tid_i] = r
        sets.append(ids)
    common = sets[0].copy()
    for s in sets[1:]:
        common &= s
    return [rows_by_id[i] for i in sorted(common) if i in rows_by_id]


def resolve_seeds_mixed(
    client: RatingClient,
    lines: list[str],
    date_end_after: str | None,
) -> tuple[
    dict[int, dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    list[int],
    list[dict[str, Any]],
]:
    """Digit-only lines = tournament id; others = name search after trimming leading punctuation.

    Name search: strip leading non-alphanumeric, then GET /tournaments?name[]=full string.
    If that returns nothing and there are at least two whitespace-separated tokens, run one
    search per word and intersect by tournament id (AND of substring matches).
    """
    seed_meta: dict[int, dict[str, Any]] = {}
    matches_by_line: dict[str, list[dict[str, Any]]] = {}
    order: list[int] = []
    resolution_warnings: list[dict[str, Any]] = []

    for line in lines:
        if tournament_line_is_seed_id(line):
            tid_raw = int(line)
            row = client.fetch_tournament_item(tid_raw)
            if row is None or row.get("id") is None:
                matches_by_line[line] = []
                continue
            tid = int(row["id"])
            matches_by_line[line] = [{"id": tid, "name": row.get("name")}]
            if tid not in seed_meta:
                seed_meta[tid] = {
                    "id": tid,
                    "name": row.get("name"),
                    "source_substrings": [],
                    "editor_surnames": editor_surnames_from_tournament(row),
                    "difficultyForecast": row.get("difficultyForecast"),
                }
                order.append(tid)
            if line not in seed_meta[tid]["source_substrings"]:
                seed_meta[tid]["source_substrings"].append(line)
        else:
            normalized = strip_leading_until_alnum(line)
            if not normalized:
                matches_by_line[line] = []
                continue
            rows = client.fetch_tournaments_by_name(normalized, date_end_after)
            words = [w for w in normalized.split() if w]
            if not rows and len(words) >= 2:
                rows = tournaments_matching_all_words(client, words, date_end_after)
                if rows:
                    resolution_warnings.append(
                        {
                            "type": "name_search_word_intersection",
                            "substring": line,
                            "normalized": normalized,
                            "words": words,
                        }
                    )
            matches_by_line[line] = [{"id": r.get("id"), "name": r.get("name")} for r in rows]
            for trow in rows:
                tid = trow.get("id")
                if tid is None:
                    continue
                tid = int(tid)
                if tid not in seed_meta:
                    seed_meta[tid] = {
                        "id": tid,
                        "name": trow.get("name"),
                        "source_substrings": [],
                        "editor_surnames": editor_surnames_from_tournament(trow),
                        "difficultyForecast": trow.get("difficultyForecast"),
                    }
                    order.append(tid)
                if line not in seed_meta[tid]["source_substrings"]:
                    seed_meta[tid]["source_substrings"].append(line)

    return seed_meta, matches_by_line, order, resolution_warnings


def short_label_from_status(status: str) -> str:
    if status in ("played_listed", "played_via_intersection"):
        return "played"
    if status == "clear":
        return "clear"
    return status


def build_summary(
    line_keys: list[str],
    matches_by_line: dict[str, list[dict[str, Any]]],
    tournaments_out: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    status_by_id: dict[int, str] = {}
    extra_by_id: dict[int, dict[str, Any]] = {}
    for row in tournaments_out:
        tid = row.get("id")
        if tid is None:
            continue
        tid = int(tid)
        status_by_id[tid] = short_label_from_status(str(row.get("status", "")))
        extra_by_id[tid] = {
            "editor_surnames": list(row.get("editor_surnames") or []),
            "difficultyForecast": row.get("difficultyForecast"),
        }

    summary: list[dict[str, Any]] = []
    for key in line_keys:
        matches = matches_by_line.get(key, [])
        if not matches:
            summary.append(
                {
                    "id": None,
                    "name": key,
                    "status": "not found",
                    "editor_surnames": [],
                    "difficultyForecast": None,
                }
            )
            continue
        for m in matches:
            tid = m.get("id")
            if tid is None:
                continue
            tid = int(tid)
            name = m.get("name")
            display = str(name).strip() if name else f"(tournament id {tid})"
            ex = extra_by_id.get(
                tid,
                {"editor_surnames": [], "difficultyForecast": None},
            )
            summary.append(
                {
                    "id": tid,
                    "name": display,
                    "status": status_by_id.get(tid, "clear"),
                    "editor_surnames": ex["editor_surnames"],
                    "difficultyForecast": ex["difficultyForecast"],
                }
            )
    return summary


def _tsv_cell(value: Any) -> str:
    s = "" if value is None else str(value)
    return s.replace("\t", " ").replace("\n", " ").replace("\r", " ")


def format_summary_lines(summary: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for row in summary:
        tid = row.get("id")
        id_s = "" if tid is None else str(tid)
        eds = row.get("editor_surnames") or []
        ed_s = "; ".join(_tsv_cell(x) for x in eds)
        df = row.get("difficultyForecast")
        df_s = "" if df is None else _tsv_cell(df)
        lines.append(
            f"{id_s}\t{row['status']}\t{_tsv_cell(row['name'])}\t({ed_s})\t{df_s}"
        )
    return lines


def build_warnings(matches_by_line: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    for sub, matches in matches_by_line.items():
        if len(matches) > 1:
            warnings.append(
                {
                    "type": "ambiguous_name_match",
                    "substring": sub,
                    "count": len(matches),
                    "tournaments": matches,
                }
            )
        if len(matches) == 0:
            warnings.append(
                {
                    "type": "no_match",
                    "substring": sub,
                    "tournaments": [],
                }
            )
    return warnings


def run_check(
    player_ids: list[int],
    tournament_lines: list[str],
    *,
    base_url: str = DEFAULT_BASE,
    date_end_after: str | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run overlap check using /players/{id}/tournaments; same report shape as tourn_check_web.run_check."""
    timing = os.environ.get("TOURN_CHECK_TIMING", "").lower() in ("1", "true", "yes")
    t_mark = time.perf_counter()

    def _timing_note(label: str) -> None:
        nonlocal t_mark
        if not timing:
            return
        now = time.perf_counter()
        print(f"[tourn_check_by_player timing] {label}: {now - t_mark:.3f}s", file=sys.stderr)
        t_mark = now

    team_ids = set(player_ids)
    client = RatingClient(base_url, verbose=verbose)
    timeout = client.timeout

    parallel = os.environ.get("TOURN_CHECK_PARALLEL", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    try:
        parallel_workers = max(1, min(32, int(os.environ.get("TOURN_CHECK_PARALLEL_WORKERS", "8"))))
    except ValueError:
        parallel_workers = 8

    seed_meta, matches_by_line, seed_order, resolution_warnings = resolve_seeds_mixed(
        client, tournament_lines, date_end_after
    )
    _timing_note("resolve seeds (name/id lookups)")

    intersections_by_seed: dict[str, list[int]] = {}
    all_tournament_ids: set[int] = set(seed_meta.keys())

    if parallel and len(seed_order) > 1:

        def _inter_job(tid: int) -> tuple[int, list[int]]:
            c = _thread_local_client(base_url, timeout, verbose)
            rows = c.fetch_intersections(tid)
            return tid, intersection_ids_from_response(rows)

        with ThreadPoolExecutor(max_workers=parallel_workers) as pool:
            futures = [pool.submit(_inter_job, tid) for tid in seed_order]
            for fut in as_completed(futures):
                tid, iids = fut.result()
                intersections_by_seed[str(tid)] = iids
                all_tournament_ids.update(iids)
    else:
        for tid in seed_order:
            inter_rows = client.fetch_intersections(tid)
            iids = intersection_ids_from_response(inter_rows)
            intersections_by_seed[str(tid)] = iids
            all_tournament_ids.update(iids)
    _timing_note(f"fetch intersections ({len(seed_order)} seeds → {len(all_tournament_ids)} tournament ids total)")

    unique_player_ids = list(dict.fromkeys(player_ids))
    tournaments_per_player: dict[int, set[int]] = {}

    if parallel and len(unique_player_ids) > 1:

        def _pt_job(pid: int) -> tuple[int, set[int]]:
            c = _thread_local_client(base_url, timeout, verbose)
            rows = c.fetch_player_tournaments(pid)
            tids: set[int] = set()
            for r in rows:
                x = tournament_id_from_player_row(r)
                if x is not None:
                    tids.add(x)
            return pid, tids

        with ThreadPoolExecutor(max_workers=parallel_workers) as pool:
            futures = [pool.submit(_pt_job, pid) for pid in unique_player_ids]
            for fut in as_completed(futures):
                pid, tids = fut.result()
                tournaments_per_player[pid] = tids
    else:
        for pid in unique_player_ids:
            rows = client.fetch_player_tournaments(pid)
            tids = set()
            for r in rows:
                x = tournament_id_from_player_row(r)
                if x is not None:
                    tids.add(x)
            tournaments_per_player[pid] = tids

    n_pt_rows = sum(len(tournaments_per_player[p]) for p in unique_player_ids)
    _timing_note(
        f"fetch player tournaments ({len(unique_player_ids)} players, {n_pt_rows} idtournament rows total)"
    )

    tournaments_out: list[dict[str, Any]] = []
    for tid in seed_order:
        meta = seed_meta[tid]
        iids = intersections_by_seed[str(tid)]
        listed_hits = sorted(p for p in team_ids if tid in tournaments_per_player.get(p, set()))
        played_listed = bool(listed_hits)

        matching_intersection_ids: list[int] = []
        inter_hits: dict[str, list[int]] = {}
        for iid in iids:
            hp = sorted(pl for pl in team_ids if iid in tournaments_per_player.get(pl, set()))
            if hp:
                matching_intersection_ids.append(iid)
                inter_hits[str(iid)] = hp

        played_intersection = bool(matching_intersection_ids)
        if played_listed:
            status = "played_listed"
        elif played_intersection:
            status = "played_via_intersection"
        else:
            status = "clear"

        tournaments_out.append(
            {
                "id": tid,
                "name": meta.get("name"),
                "source_substrings": meta.get("source_substrings", []),
                "editor_surnames": meta.get("editor_surnames", []),
                "difficultyForecast": meta.get("difficultyForecast"),
                "intersection_ids": iids,
                "played_listed": played_listed,
                "played_intersection": played_intersection,
                "status": status,
                "matching_players_listed": listed_hits,
                "matching_players_by_intersection_id": inter_hits,
                "matching_intersection_ids": matching_intersection_ids,
            }
        )

    summary = build_summary(tournament_lines, matches_by_line, tournaments_out)
    kinds = ["id" if tournament_line_is_seed_id(ln) else "name" for ln in tournament_lines]
    _timing_note("build summary / report dict")

    return {
        "input": {
            "player_ids": player_ids,
            "tournament_lines": tournament_lines,
            "tournament_line_kinds": kinds,
            "date_end_strictly_after": date_end_after,
            "base_url": base_url,
            "overlap_strategy": "player_tournaments_api",
        },
        "intersections_by_seed": intersections_by_seed,
        "tournaments": tournaments_out,
        "warnings": resolution_warnings + build_warnings(matches_by_line),
        "summary": summary,
    }


def report_to_json(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2) + "\n"
