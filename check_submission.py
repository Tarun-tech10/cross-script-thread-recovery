#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate submission.csv against the rules in the challenge brief.

    python3 check_submission.py [submission.csv] [data_dir]

Checks, in order:
  1. exactly two columns, named query_id and conversation_id
  2. one row per query_id in test_queries.csv, none missing, none extra, no dupes
  3. every conversation_id is a non-empty string
  4. no predicted conversation spans two pages (a label reused across pages is
     allowed by the metric, but it is worth knowing when it happens)
and then prints the shape of the predicted clustering next to the numbers the
brief quotes for the test set (2,301 true conversations over 9,921 messages).
"""
import sys
import os
import pandas as pd


def main(argv):
    sub_path = argv[1] if len(argv) > 1 else "submission.csv"
    data_dir = argv[2] if len(argv) > 2 else "CST_DATA"
    if not os.path.isdir(data_dir):
        for c in ["../CST_DATA", "CST_DATA", ".", ".."]:
            if os.path.isfile(os.path.join(c, "test_queries.csv")):
                data_dir = c
                break

    sub = pd.read_csv(sub_path, dtype=str)
    q = pd.read_csv(os.path.join(data_dir, "test_queries.csv"), dtype=str)
    fail = []

    if list(sub.columns) != ["query_id", "conversation_id"]:
        fail.append("columns are %s, expected ['query_id', 'conversation_id']"
                    % list(sub.columns))
    if sub["query_id"].duplicated().any():
        fail.append("%d duplicated query_id values" % int(sub["query_id"].duplicated().sum()))

    missing = set(q["query_id"]) - set(sub["query_id"])
    extra = set(sub["query_id"]) - set(q["query_id"])
    if missing:
        fail.append("%d query_id values missing (e.g. %s)"
                    % (len(missing), sorted(missing)[:3]))
    if extra:
        fail.append("%d unknown query_id values (e.g. %s)"
                    % (len(extra), sorted(extra)[:3]))
    if sub["conversation_id"].isna().any() or (sub["conversation_id"].fillna("").str.len() == 0).any():
        fail.append("some conversation_id values are empty")

    merged = q.merge(sub, on="query_id", how="left")
    spans = merged.groupby("conversation_id")["file_id"].nunique()
    crossing = int((spans > 1).sum())

    n_conv = int(merged.groupby("file_id")["conversation_id"].nunique().sum())
    sizes = merged.groupby(["file_id", "conversation_id"]).size()

    print("rows            : %d (queries: %d)" % (len(sub), len(q)))
    print("conversations   : %d   [brief: 2301 true conversations]" % n_conv)
    print("pages           : %d   [brief: 47 test pages]" % merged["file_id"].nunique())
    print("mean size       : %.2f   [brief implies 9921/2301 = 4.31]" % sizes.mean())
    print("singletons      : %d (%.1f%%)" % (int((sizes == 1).sum()), 100.0 * (sizes == 1).mean()))
    print("labels reused across pages: %d (harmless -- the metric compares within a page)"
          % crossing)

    if fail:
        print("\nFAILED:")
        for f in fail:
            print("  - " + f)
        return 1
    print("\nOK - submission is valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
