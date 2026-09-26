#!/usr/bin/env python3
"""Independently bind global product gates to their complete live evidence."""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_m018_global_corrections as producer  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_EXECUTION_IDS = (
    "reverb::gcc13::normal::baseline",
    "reverb::gcc13::normal::early-mix",
    "reverb::gcc13::normal::early-predelay",
    "reverb::gcc13::normal::partial-shaping-publication",
    "reverb::gcc13::sanitizer::baseline",
    "reverb::gcc13::sanitizer::early-mix",
    "reverb::gcc13::sanitizer::early-predelay",
    "reverb::gcc13::sanitizer::partial-shaping-publication",
    "reverb::clang18::normal::baseline",
    "reverb::clang18::normal::early-mix",
    "reverb::clang18::normal::early-predelay",
    "reverb::clang18::normal::partial-shaping-publication",
    "reverb::clang18::sanitizer::baseline",
    "reverb::clang18::sanitizer::early-mix",
    "reverb::clang18::sanitizer::early-predelay",
    "reverb::clang18::sanitizer::partial-shaping-publication",
)
EXPECTED_EXECUTION_CARDINALITY = 16
EXPECTED_ROWS = (
    ("reverb::gcc13::normal::baseline", "gcc13", "normal", "baseline", "GREEN"),
    ("reverb::gcc13::normal::early-mix", "gcc13", "normal", "early-mix", "EXPECTED_RED"),
    ("reverb::gcc13::normal::early-predelay", "gcc13", "normal", "early-predelay", "EXPECTED_RED"),
    ("reverb::gcc13::normal::partial-shaping-publication", "gcc13", "normal", "partial-shaping-publication", "EXPECTED_RED"),
    ("reverb::gcc13::sanitizer::baseline", "gcc13", "sanitizer", "baseline", "GREEN"),
    ("reverb::gcc13::sanitizer::early-mix", "gcc13", "sanitizer", "early-mix", "EXPECTED_RED"),
    ("reverb::gcc13::sanitizer::early-predelay", "gcc13", "sanitizer", "early-predelay", "EXPECTED_RED"),
    ("reverb::gcc13::sanitizer::partial-shaping-publication", "gcc13", "sanitizer", "partial-shaping-publication", "EXPECTED_RED"),
    ("reverb::clang18::normal::baseline", "clang18", "normal", "baseline", "GREEN"),
    ("reverb::clang18::normal::early-mix", "clang18", "normal", "early-mix", "EXPECTED_RED"),
    ("reverb::clang18::normal::early-predelay", "clang18", "normal", "early-predelay", "EXPECTED_RED"),
    ("reverb::clang18::normal::partial-shaping-publication", "clang18", "normal", "partial-shaping-publication", "EXPECTED_RED"),
    ("reverb::clang18::sanitizer::baseline", "clang18", "sanitizer", "baseline", "GREEN"),
    ("reverb::clang18::sanitizer::early-mix", "clang18", "sanitizer", "early-mix", "EXPECTED_RED"),
    ("reverb::clang18::sanitizer::early-predelay", "clang18", "sanitizer", "early-predelay", "EXPECTED_RED"),
    ("reverb::clang18::sanitizer::partial-shaping-publication", "clang18", "sanitizer", "partial-shaping-publication", "EXPECTED_RED"),
)
EXPECTED_SOURCE_HASHES = {
    "baseline": "294a1053756adc86d84ab22240afdc4a55a706bef5e09a890ba6e084f4f6a80f",
    "early-mix": "a92a17baa0dee3618acf0a0de08a2e70b74fe479c73fd0bbf53c7dde4ec6ccfa",
    "early-predelay": "2bfc9fc8137ea9c214c740e0e953b4c1f9a5e13b096c5fcd0a4967c64cca23b6",
    "partial-shaping-publication": "1f6af93ae51eeaead1cb4c929641ecaa7817d687b36ee7ff7fb6441cce6b2010",
}
EXPECTED_INSERTION_HASHES = {
    "baseline": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "early-mix": "198cbf40437bae2cf942bb530552687078544ce62a5a38ee3fb1b1977a256b57",
    "early-predelay": "6bbfb61f9aa7c7d7f9502d11ccbd4b3907318e1f9d1e4f8c0a9fc606389d6936",
    "partial-shaping-publication": "4d3adddfcbc82e7bd4e6ee7a4015289675870106876426bcea60fd400166e87f",
}
EXPECTED_PATCH_HASHES = {
    "baseline": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "early-mix": "39400ede7f92730ec68e02599d8d0c125e066b571573422d036aaefaaa3222e9",
    "early-predelay": "42f1bb61861e677382493a2675f5d387959dab2af11aabca81bc1572ce977aff",
    "partial-shaping-publication": "557fbc4018cef7d9f91ffc8c9aee972a0b7218e24b210238f25f33834a8fdadc",
}
BASELINE_STDOUT = (
    "PASS reverb-slot-state-machine\n"
    "PASS reverb-fixed-audio-operation-bound\n"
    "PASS reverb-publisher-phase-parking\n"
    "PASS reverb-no-audio-allocation-or-final-release\n"
    "PASS reverb-premature-reuse-mutant\n"
    "PASS reverb-publisher-starvation-mutant\n"
    "PASS reverb-generation-aba\n"
    "PASS reverb-getconvolver-pin-lifetime\n"
    "PASS reverb-reset-metadata-exception-shutdown\n"
    "PASS reverb-retained-bank-memory-bound\n"
    "PASS reverb-race-sanitizer-matrix\n"
    "11 checks, 0 failures\n"
)
MUTANT_STDOUT = (
    "PASS reverb-slot-state-machine\n"
    "PASS reverb-fixed-audio-operation-bound\n"
    "PASS reverb-publisher-phase-parking\n"
    "PASS reverb-no-audio-allocation-or-final-release\n"
    "PASS reverb-premature-reuse-mutant\n"
    "PASS reverb-publisher-starvation-mutant\n"
    "PASS reverb-generation-aba\n"
    "PASS reverb-getconvolver-pin-lifetime\n"
    "PASS reverb-retained-bank-memory-bound\n"
    "PASS reverb-race-sanitizer-matrix\n"
    "11 checks, 1 failures\n"
)
MUTANT_STDERR = (
    "FAIL reverb-reset-metadata-exception-shutdown: "
    "failed setState changed serialized state bytes\n"
)
EXPECTED_OUTPUT_CONTRACTS = {
    "GREEN": (0, BASELINE_STDOUT, ""),
    "EXPECTED_RED": (1, MUTANT_STDOUT, MUTANT_STDERR),
}
SANITIZER_DIAGNOSTICS = (
    "ERROR: AddressSanitizer",
    "runtime error:",
    "LeakSanitizer",
)
COMMON_COMPILE_FLAGS = (
    "-std=c++20", "-O1", "-g", "-Wall", "-Wextra", "-Wpedantic",
    "-Werror", "-DDSPARK_REVERB_TEST_GENERATION_MAX=1", "-pthread",
)
SANITIZER_FLAGS = (
    "-fno-omit-frame-pointer",
    "-fsanitize=address,undefined,float-cast-overflow",
    "-fno-sanitize-recover=all",
)
REMOVED_RUNTIME_ENVIRONMENT = (
    "ASAN_OPTIONS", "UBSAN_OPTIONS", "LSAN_OPTIONS",
)
NORMAL_RUNTIME_ENVIRONMENT = {"LC_ALL": "C", "LANG": "C"}
SANITIZER_RUNTIME_ENVIRONMENT = {
    "LC_ALL": "C",
    "LANG": "C",
    "ASAN_OPTIONS": "halt_on_error=1:abort_on_error=1:detect_leaks=1",
    "UBSAN_OPTIONS": "halt_on_error=1:print_stacktrace=1",
}
EXPECTED_OUTPUTS = (
    "c1-stationary-fidelity.csv",
    "c2-spurious-floor.csv",
    "c3-transient-preservation.csv",
    "c4-vertical-coherence.csv",
    "c5-stereo-integrity.csv",
    "c6-exact-ratio-drift.csv",
    "c7-unity-passthrough.csv",
    "c8-chopping-determinism.csv",
    "timestretch-metrics.md",
)
TIMESTRETCH_TRANSACTION_DIRECTORY = ".dspark-timestretch-transaction"
TIMESTRETCH_FAILURE_PHASES = (
    ("STAGE", "DSPARK_TIMESTRETCH_FAIL_STAGE_INDEX"),
    ("COMMIT", "DSPARK_TIMESTRETCH_FAIL_COMMIT_INDEX"),
)
TIMESTRETCH_BUILD_LANES = (
    ("gcc13", "normal"),
    ("gcc13", "sanitizer"),
    ("clang18", "normal"),
    ("clang18", "sanitizer"),
)
TIMESTRETCH_TRANSACTION_STATES = (
    "EMPTY", "SETUP", "STAGING", "STAGED", "BACKUP_IN_PROGRESS",
    "PUBLISH_IN_PROGRESS", "COMMIT_READY", "ROLLBACK_REQUIRED",
    "RECOVERY_REQUIRED_ROLLBACK", "COMMITTED_CLEANUP_PENDING",
    "RECOVERY_REQUIRED_CLEANUP", "COMPLETE",
    "FOREIGN_OR_MALFORMED_COLLISION",
)
TIMESTRETCH_TRANSACTION_TRANSITIONS = (
    ("EMPTY", "exclusive-directory-and-initial-journal", "SETUP"),
    ("SETUP", "open-first-fixed-stage-part", "STAGING"),
    ("STAGING", "finish-stage-index", "STAGING"),
    ("STAGING", "finish-stage-index-8", "STAGED"),
    ("STAGED", "record-backup-intent", "BACKUP_IN_PROGRESS"),
    ("BACKUP_IN_PROGRESS", "rename-old-final-to-backup",
     "BACKUP_IN_PROGRESS"),
    ("BACKUP_IN_PROGRESS", "all-old-rows-reconciled",
     "PUBLISH_IN_PROGRESS"),
    ("PUBLISH_IN_PROGRESS", "rename-staged-new-to-final",
     "PUBLISH_IN_PROGRESS"),
    ("PUBLISH_IN_PROGRESS", "all-nine-new-finals-validated", "COMMIT_READY"),
    ("COMMIT_READY", "promote-committed-journal-generation",
     "COMMITTED_CLEANUP_PENDING"),
    ("SETUP", "precommit-failure-or-recovery", "ROLLBACK_REQUIRED"),
    ("STAGING", "precommit-failure-or-recovery", "ROLLBACK_REQUIRED"),
    ("STAGED", "precommit-failure-or-recovery", "ROLLBACK_REQUIRED"),
    ("BACKUP_IN_PROGRESS", "precommit-failure-or-recovery",
     "ROLLBACK_REQUIRED"),
    ("PUBLISH_IN_PROGRESS", "precommit-failure-or-recovery",
     "ROLLBACK_REQUIRED"),
    ("COMMIT_READY", "commit-record-failure-or-recovery",
     "ROLLBACK_REQUIRED"),
    ("ROLLBACK_REQUIRED", "remove-provisional-new-index",
     "ROLLBACK_REQUIRED"),
    ("ROLLBACK_REQUIRED", "restore-old-backup-index", "ROLLBACK_REQUIRED"),
    ("ROLLBACK_REQUIRED", "rollback-operation-failed",
     "RECOVERY_REQUIRED_ROLLBACK"),
    ("RECOVERY_REQUIRED_ROLLBACK", "retry-rollback",
     "RECOVERY_REQUIRED_ROLLBACK"),
    ("RECOVERY_REQUIRED_ROLLBACK", "retry-rollback-complete", "EMPTY"),
    ("ROLLBACK_REQUIRED", "rollback-complete", "EMPTY"),
    ("COMMITTED_CLEANUP_PENDING", "remove-old-backup-index",
     "COMMITTED_CLEANUP_PENDING"),
    ("COMMITTED_CLEANUP_PENDING", "cleanup-failed",
     "RECOVERY_REQUIRED_CLEANUP"),
    ("RECOVERY_REQUIRED_CLEANUP", "retry-cleanup",
     "RECOVERY_REQUIRED_CLEANUP"),
    ("RECOVERY_REQUIRED_CLEANUP", "retry-cleanup-complete", "COMPLETE"),
    ("COMMITTED_CLEANUP_PENDING", "cleanup-complete", "COMPLETE"),
    ("ANY_WITH_COLLISION", "invalid-foreign-or-ambiguous-journal",
     "FOREIGN_OR_MALFORMED_COLLISION"),
)
TIMESTRETCH_JOURNAL_POINTS = (
    *("setup:{:02d}".format(index) for index in range(3)),
    *("stage:{:02d}".format(index) for index in range(9)),
    *("backup:{:02d}".format(index) for index in range(9)),
    *("publish:{:02d}".format(index) for index in range(9)),
    *("rollback-remove:{:02d}".format(index) for index in range(9)),
    *("rollback-restore:{:02d}".format(index) for index in range(9)),
    *("recovery-reconcile:{:02d}".format(index) for index in range(9)),
    "commit-marker",
    *("cleanup-backup:{:02d}".format(index) for index in range(9)),
    *("finalize:{:02d}".format(index) for index in range(3)),
)
TIMESTRETCH_DURABLE_FAULT_POINTS = (
    'setup:00',
    'setup:01',
    'setup:02',
    'journal-write:setup:00',
    'journal-write:setup:01',
    'journal-write:setup:02',
    'journal-write:stage:00',
    'journal-write:stage:01',
    'journal-write:stage:02',
    'journal-write:stage:03',
    'journal-write:stage:04',
    'journal-write:stage:05',
    'journal-write:stage:06',
    'journal-write:stage:07',
    'journal-write:stage:08',
    'journal-write:backup:00',
    'journal-write:backup:01',
    'journal-write:backup:02',
    'journal-write:backup:03',
    'journal-write:backup:04',
    'journal-write:backup:05',
    'journal-write:backup:06',
    'journal-write:backup:07',
    'journal-write:backup:08',
    'journal-write:publish:00',
    'journal-write:publish:01',
    'journal-write:publish:02',
    'journal-write:publish:03',
    'journal-write:publish:04',
    'journal-write:publish:05',
    'journal-write:publish:06',
    'journal-write:publish:07',
    'journal-write:publish:08',
    'journal-write:rollback-remove:00',
    'journal-write:rollback-remove:01',
    'journal-write:rollback-remove:02',
    'journal-write:rollback-remove:03',
    'journal-write:rollback-remove:04',
    'journal-write:rollback-remove:05',
    'journal-write:rollback-remove:06',
    'journal-write:rollback-remove:07',
    'journal-write:rollback-remove:08',
    'journal-write:rollback-restore:00',
    'journal-write:rollback-restore:01',
    'journal-write:rollback-restore:02',
    'journal-write:rollback-restore:03',
    'journal-write:rollback-restore:04',
    'journal-write:rollback-restore:05',
    'journal-write:rollback-restore:06',
    'journal-write:rollback-restore:07',
    'journal-write:rollback-restore:08',
    'journal-write:recovery-reconcile:00',
    'journal-write:recovery-reconcile:01',
    'journal-write:recovery-reconcile:02',
    'journal-write:recovery-reconcile:03',
    'journal-write:recovery-reconcile:04',
    'journal-write:recovery-reconcile:05',
    'journal-write:recovery-reconcile:06',
    'journal-write:recovery-reconcile:07',
    'journal-write:recovery-reconcile:08',
    'journal-write:commit-marker',
    'journal-write:cleanup-backup:00',
    'journal-write:cleanup-backup:01',
    'journal-write:cleanup-backup:02',
    'journal-write:cleanup-backup:03',
    'journal-write:cleanup-backup:04',
    'journal-write:cleanup-backup:05',
    'journal-write:cleanup-backup:06',
    'journal-write:cleanup-backup:07',
    'journal-write:cleanup-backup:08',
    'journal-write:finalize:00',
    'journal-write:finalize:01',
    'journal-write:finalize:02',
    'journal-replace:setup:00',
    'journal-replace:setup:01',
    'journal-replace:setup:02',
    'journal-replace:stage:00',
    'journal-replace:stage:01',
    'journal-replace:stage:02',
    'journal-replace:stage:03',
    'journal-replace:stage:04',
    'journal-replace:stage:05',
    'journal-replace:stage:06',
    'journal-replace:stage:07',
    'journal-replace:stage:08',
    'journal-replace:backup:00',
    'journal-replace:backup:01',
    'journal-replace:backup:02',
    'journal-replace:backup:03',
    'journal-replace:backup:04',
    'journal-replace:backup:05',
    'journal-replace:backup:06',
    'journal-replace:backup:07',
    'journal-replace:backup:08',
    'journal-replace:publish:00',
    'journal-replace:publish:01',
    'journal-replace:publish:02',
    'journal-replace:publish:03',
    'journal-replace:publish:04',
    'journal-replace:publish:05',
    'journal-replace:publish:06',
    'journal-replace:publish:07',
    'journal-replace:publish:08',
    'journal-replace:rollback-remove:00',
    'journal-replace:rollback-remove:01',
    'journal-replace:rollback-remove:02',
    'journal-replace:rollback-remove:03',
    'journal-replace:rollback-remove:04',
    'journal-replace:rollback-remove:05',
    'journal-replace:rollback-remove:06',
    'journal-replace:rollback-remove:07',
    'journal-replace:rollback-remove:08',
    'journal-replace:rollback-restore:00',
    'journal-replace:rollback-restore:01',
    'journal-replace:rollback-restore:02',
    'journal-replace:rollback-restore:03',
    'journal-replace:rollback-restore:04',
    'journal-replace:rollback-restore:05',
    'journal-replace:rollback-restore:06',
    'journal-replace:rollback-restore:07',
    'journal-replace:rollback-restore:08',
    'journal-replace:recovery-reconcile:00',
    'journal-replace:recovery-reconcile:01',
    'journal-replace:recovery-reconcile:02',
    'journal-replace:recovery-reconcile:03',
    'journal-replace:recovery-reconcile:04',
    'journal-replace:recovery-reconcile:05',
    'journal-replace:recovery-reconcile:06',
    'journal-replace:recovery-reconcile:07',
    'journal-replace:recovery-reconcile:08',
    'journal-replace:commit-marker',
    'journal-replace:cleanup-backup:00',
    'journal-replace:cleanup-backup:01',
    'journal-replace:cleanup-backup:02',
    'journal-replace:cleanup-backup:03',
    'journal-replace:cleanup-backup:04',
    'journal-replace:cleanup-backup:05',
    'journal-replace:cleanup-backup:06',
    'journal-replace:cleanup-backup:07',
    'journal-replace:cleanup-backup:08',
    'journal-replace:finalize:00',
    'journal-replace:finalize:01',
    'journal-replace:finalize:02',
    'backup-rename:00',
    'backup-rename:01',
    'backup-rename:02',
    'backup-rename:03',
    'backup-rename:04',
    'backup-rename:05',
    'backup-rename:06',
    'backup-rename:07',
    'backup-rename:08',
    'publish-rename:00',
    'publish-rename:01',
    'publish-rename:02',
    'publish-rename:03',
    'publish-rename:04',
    'publish-rename:05',
    'publish-rename:06',
    'publish-rename:07',
    'publish-rename:08',
    'new-final-remove:00',
    'new-final-remove:01',
    'new-final-remove:02',
    'new-final-remove:03',
    'new-final-remove:04',
    'new-final-remove:05',
    'new-final-remove:06',
    'new-final-remove:07',
    'new-final-remove:08',
    'restore-rename:00',
    'restore-rename:01',
    'restore-rename:02',
    'restore-rename:03',
    'restore-rename:04',
    'restore-rename:05',
    'restore-rename:06',
    'restore-rename:07',
    'restore-rename:08',
    'backup-cleanup:00',
    'backup-cleanup:01',
    'backup-cleanup:02',
    'backup-cleanup:03',
    'backup-cleanup:04',
    'backup-cleanup:05',
    'backup-cleanup:06',
    'backup-cleanup:07',
    'backup-cleanup:08',
    'journal-cleanup:00',
    'journal-cleanup:01',
    'directory-cleanup:00',
    'directory-cleanup:01',
    'directory-cleanup:02',
)
TIMESTRETCH_CRASH_POINTS = (
    *("setup:{:02d}".format(index) for index in range(3)),
    *("stage:{:02d}".format(index) for index in range(9)),
    "staged-boundary",
    *(point for index in range(9) for point in (
        "backup-before:{:02d}".format(index),
        "backup-after:{:02d}".format(index))),
    *(point for index in range(9) for point in (
        "publish-before:{:02d}".format(index),
        "publish-after:{:02d}".format(index))),
    *(point for index in range(9) for point in (
        "rollback-remove-before:{:02d}".format(index),
        "rollback-remove-after:{:02d}".format(index))),
    *(point for index in range(9) for point in (
        "rollback-restore-before:{:02d}".format(index),
        "rollback-restore-after:{:02d}".format(index))),
    *(point for index in range(9) for point in (
        "cleanup-backup-before:{:02d}".format(index),
        "cleanup-backup-after:{:02d}".format(index))),
    "commit-ready",
    "post-commit-marker",
    *(point for index in range(2) for point in (
        "journal-cleanup-before:{:02d}".format(index),
        "journal-cleanup-after:{:02d}".format(index))),
    *(point for index in range(3) for point in (
        "directory-cleanup-before:{:02d}".format(index),
        "directory-cleanup-after:{:02d}".format(index))),
)
TIMESTRETCH_JOURNAL_MUTATIONS = (
    "truncation",
    "oversize",
    "duplicate-key",
    "unknown-key",
    "unknown-output-name",
    "path-escape-name",
    "checksum-mismatch",
    "generation-gap",
    "impossible-state-combination",
    "symlink-substitution",
)
TIMESTRETCH_OUTER_MUTANTS = (
    "delete-durable-recovery-invocation",
    "delete-literal-fault-inventory",
    "replace-literal-inventory-with-textual-marker",
    "delete-observed-versus-expected-identity-comparison",
    "delete-journal-mutation-loop",
    "delete-crash-cut-loop",
    "delete-ordinary-success-suppression-oracle",
    "delete-recovery-terminal-oracle",
    "delete-inherited-legacy-loop",
    "delete-ctest-binding",
    "delete-ci-binding",
    "delete-docs-binding",
)
TIMESTRETCH_OUTER_MUTANT_POINTS = (
    "delete durable recovery invocation",
    "delete literal fault inventory",
    "replace literal inventory with textual marker",
    "delete observed-versus-expected identity comparison",
    "delete journal mutation loop",
    "delete crash-cut loop",
    "delete ordinary-success suppression oracle",
    "delete recovery-terminal oracle",
    "delete inherited legacy loop",
    "delete CTest binding",
    "delete CI binding",
    "delete docs binding",
)
TIMESTRETCH_COHERENCE_TRANSITIONS = (
    ("T01", "SETUP", "SETUP", "SETUP_PROGRESS"),
    ("T02", "SETUP", "STAGING", "STAGE_ESTABLISH_FIRST"),
    ("T03", "STAGING", "STAGING", "STAGE_ESTABLISH_MORE"),
    ("T04", "STAGING", "STAGED", "STAGE_ESTABLISH_LAST"),
    ("T05", "STAGED", "BACKUP_IN_PROGRESS", "BACKUP_INTENT_FIRST"),
    ("T06", "BACKUP_IN_PROGRESS", "BACKUP_IN_PROGRESS", "BACKUP_APPLIED"),
    ("T07", "BACKUP_IN_PROGRESS", "BACKUP_IN_PROGRESS", "BACKUP_INTENT_NEXT"),
    ("T08", "BACKUP_IN_PROGRESS", "PUBLISH_IN_PROGRESS", "PUBLISH_INTENT_FIRST"),
    ("T09", "PUBLISH_IN_PROGRESS", "PUBLISH_IN_PROGRESS", "PUBLISH_APPLIED"),
    ("T10", "PUBLISH_IN_PROGRESS", "PUBLISH_IN_PROGRESS", "PUBLISH_INTENT_NEXT"),
    ("T11", "PUBLISH_IN_PROGRESS", "COMMIT_READY", "COMMIT_READY"),
    ("T12", "COMMIT_READY", "COMMITTED_CLEANUP_PENDING", "COMMIT_MARKER"),
    ("T13", "SETUP", "ROLLBACK_REQUIRED", "ENTER_ROLLBACK"),
    ("T14", "STAGING", "ROLLBACK_REQUIRED", "ENTER_ROLLBACK"),
    ("T15", "STAGED", "ROLLBACK_REQUIRED", "ENTER_ROLLBACK"),
    ("T16", "BACKUP_IN_PROGRESS", "ROLLBACK_REQUIRED", "ENTER_ROLLBACK"),
    ("T17", "PUBLISH_IN_PROGRESS", "ROLLBACK_REQUIRED", "ENTER_ROLLBACK"),
    ("T18", "COMMIT_READY", "ROLLBACK_REQUIRED", "ENTER_ROLLBACK"),
    ("T19", "ROLLBACK_REQUIRED", "ROLLBACK_REQUIRED", "ROLLBACK_REMOVE_INTENT"),
    ("T20", "ROLLBACK_REQUIRED", "ROLLBACK_REQUIRED", "ROLLBACK_REMOVE_APPLIED"),
    ("T21", "ROLLBACK_REQUIRED", "ROLLBACK_REQUIRED", "ROLLBACK_RESTORE_INTENT"),
    ("T22", "ROLLBACK_REQUIRED", "ROLLBACK_REQUIRED", "ROLLBACK_RESTORE_APPLIED"),
    ("T23", "ROLLBACK_REQUIRED", "RECOVERY_REQUIRED_ROLLBACK", "ROLLBACK_FAILED"),
    ("T24", "RECOVERY_REQUIRED_ROLLBACK", "RECOVERY_REQUIRED_ROLLBACK", "ROLLBACK_FAILED_RETRY"),
    ("T25", "RECOVERY_REQUIRED_ROLLBACK", "ROLLBACK_REQUIRED", "RETRY_ROLLBACK"),
    ("T26", "ROLLBACK_REQUIRED", "ROLLBACK_REQUIRED", "ROLLBACK_FINALIZE_STEP"),
    ("T27", "COMMITTED_CLEANUP_PENDING", "COMMITTED_CLEANUP_PENDING", "CLEANUP_INTENT"),
    ("T28", "COMMITTED_CLEANUP_PENDING", "COMMITTED_CLEANUP_PENDING", "CLEANUP_APPLIED"),
    ("T29", "COMMITTED_CLEANUP_PENDING", "RECOVERY_REQUIRED_CLEANUP", "CLEANUP_FAILED"),
    ("T30", "RECOVERY_REQUIRED_CLEANUP", "RECOVERY_REQUIRED_CLEANUP", "CLEANUP_FAILED_RETRY"),
    ("T31", "RECOVERY_REQUIRED_CLEANUP", "COMMITTED_CLEANUP_PENDING", "RETRY_CLEANUP"),
    ("T32", "COMMITTED_CLEANUP_PENDING", "COMMITTED_CLEANUP_PENDING", "COMMITTED_FINALIZE_STEP"),
)
TIMESTRETCH_COHERENCE_PAIR_CASES = (
    "legal-single-genesis", "legal-single-precommit",
    "legal-single-committed", "legal-linked-precommit",
    "legal-linked-enter-rollback", "legal-linked-committed-cleanup",
    "legal-equal-identical", "legal-stale-lower",
    "legal-idempotent-rollback-retry",
    "legal-idempotent-cleanup-retry", "predecessor-mismatch",
    "replay-nonpredecessor", "foreign-transaction", "foreign-root",
    "generation-gap", "equal-different", "lower-generation-zero",
    "higher-generation-zero", "committed-to-setup",
    "committed-to-staging", "committed-to-staged",
    "committed-to-backup", "committed-to-publish",
    "committed-to-commit-ready", "committed-to-rollback",
    "committed-to-recovery-rollback", "generation-max-single",
    "generation-max-pair", "valid-plus-invalid-final",
    "legacy-v1-single", "legacy-v1-pair", "mixed-v1-v2",
    "linked-temp-unpromoted", "bad-temp-predecessor",
    "two-finals-plus-temp", "no-valid-slot",
)
TIMESTRETCH_COHERENCE_METADATA_CASES = (
    "transaction-id", "root-binding", "output-index", "expected-name",
    "old-present", "old-size", "old-sha256", "new-size-established",
    "new-sha256-established", "old-location", "new-location", "state",
    "pending-kind", "pending-index", "rollback-cursor", "cleanup-cursor",
    "finalize-cursor", "transition-id",
)
TIMESTRETCH_COHERENCE_OUTER_MUTANTS = (
    "delete-pair-validator", "delete-predecessor-binding",
    "delete-metadata-comparison", "delete-transition-inventory-comparison",
    "delete-execution-inventory-comparison", "delete-ctest-binding",
    "delete-ci-workflow-binding", "delete-docs-workflow-binding",
    "delete-transition-row::T01", "delete-transition-row::T02",
    "delete-transition-row::T03", "delete-transition-row::T04",
    "delete-transition-row::T05", "delete-transition-row::T06",
    "delete-transition-row::T07", "delete-transition-row::T08",
    "delete-transition-row::T09", "delete-transition-row::T10",
    "delete-transition-row::T11", "delete-transition-row::T12",
    "delete-transition-row::T13", "delete-transition-row::T14",
    "delete-transition-row::T15", "delete-transition-row::T16",
    "delete-transition-row::T17", "delete-transition-row::T18",
    "delete-transition-row::T19", "delete-transition-row::T20",
    "delete-transition-row::T21", "delete-transition-row::T22",
    "delete-transition-row::T23", "delete-transition-row::T24",
    "delete-transition-row::T25", "delete-transition-row::T26",
    "delete-transition-row::T27", "delete-transition-row::T28",
    "delete-transition-row::T29", "delete-transition-row::T30",
    "delete-transition-row::T31", "delete-transition-row::T32",
)


def expected_timestretch_execution_ids() -> tuple[str, ...]:
    rows: list[str] = []
    for compiler, profile in TIMESTRETCH_BUILD_LANES:
        lane = "{}::{}".format(compiler, profile)
        for mode in ("one-shot", "persistent"):
            rows.extend(
                "timestretch::{}::fault::{}::{}".format(lane, mode, point)
                for point in TIMESTRETCH_DURABLE_FAULT_POINTS)
        rows.extend(
            "timestretch::{}::crash::{}".format(lane, point)
            for point in TIMESTRETCH_CRASH_POINTS)
        rows.extend(
            "timestretch::{}::journal-mutation::{}".format(lane, mutation)
            for mutation in TIMESTRETCH_JOURNAL_MUTATIONS)
        rows.extend((
            "timestretch::{}::baseline::fresh".format(lane),
            "timestretch::{}::baseline::replacement".format(lane),
        ))
    for compiler in ("gcc13", "clang18"):
        for phase in ("stage", "commit"):
            rows.extend(
                "timestretch::{}::normal::legacy-{}::{:02d}".format(
                    compiler, phase, index)
                for index in range(9))
    rows.extend("timestretch::outer-mutant::" + item
                for item in TIMESTRETCH_OUTER_MUTANTS)
    for compiler, profile in TIMESTRETCH_BUILD_LANES:
        lane = "{}::{}".format(compiler, profile)
        rows.extend(
            "timestretch::{}::coherence-pair::{}".format(lane, item)
            for item in TIMESTRETCH_COHERENCE_PAIR_CASES)
        rows.extend(
            "timestretch::{}::coherence-transition::{}".format(
                lane, transition[0])
            for transition in TIMESTRETCH_COHERENCE_TRANSITIONS)
        rows.extend(
            "timestretch::{}::coherence-metadata::{}".format(lane, item)
            for item in TIMESTRETCH_COHERENCE_METADATA_CASES)
    rows.extend("timestretch::coherence-outer-mutant::" + item
                for item in TIMESTRETCH_COHERENCE_OUTER_MUTANTS)
    return tuple(rows)


def expected_timestretch_execution_rows() -> tuple[dict[str, str], ...]:
    rows: list[dict[str, str]] = []
    for compiler, profile in TIMESTRETCH_BUILD_LANES:
        lane = "{}::{}".format(compiler, profile)
        for mode in ("one-shot", "persistent"):
            for point in TIMESTRETCH_DURABLE_FAULT_POINTS:
                rows.append({
                    "class": "durable-fault",
                    "id": "timestretch::{}::fault::{}::{}".format(
                        lane, mode, point),
                    "lane": lane,
                    "mode": mode,
                    "oracle": "typed-safe-terminal-and-replay",
                    "point": point,
                })
        for point in TIMESTRETCH_CRASH_POINTS:
            rows.append({
                "class": "crash-cut",
                "id": "timestretch::{}::crash::{}".format(lane, point),
                "lane": lane,
                "mode": "process-interruption",
                "oracle": "idempotent-authority-preserving-replay",
                "point": point,
            })
        for point in TIMESTRETCH_JOURNAL_MUTATIONS:
            rows.append({
                "class": "journal-mutation",
                "id": "timestretch::{}::journal-mutation::{}".format(
                    lane, point),
                "lane": lane,
                "mode": "malformed-or-foreign",
                "oracle": "typed-collision-untouched",
                "point": point,
            })
        for point in ("fresh", "replacement"):
            rows.append({
                "class": "baseline",
                "id": "timestretch::{}::baseline::{}".format(lane, point),
                "lane": lane,
                "mode": "success",
                "oracle": "exact-nine-no-residue-one-success-line",
                "point": point,
            })
    for compiler in ("gcc13", "clang18"):
        lane = compiler + "::normal"
        for phase in ("stage", "commit"):
            for index in range(9):
                rows.append({
                    "class": "legacy-36",
                    "id": "timestretch::{}::legacy-{}::{:02d}".format(
                        lane, phase, index),
                    "lane": lane,
                    "mode": "one-shot",
                    "oracle": "P5-byte-exact-sentinel-and-zero-residue",
                    "point": "{}:{:02d}".format(phase, index),
                })
    for point, description in zip(
            TIMESTRETCH_OUTER_MUTANTS, TIMESTRETCH_OUTER_MUTANT_POINTS):
        rows.append({
            "class": "outer-mutant",
            "id": "timestretch::outer-mutant::" + point,
            "lane": "python-outer",
            "mode": "negative-control",
            "oracle": "named-nonzero-no-text-marker-substitute",
            "point": description,
        })
    for compiler, profile in TIMESTRETCH_BUILD_LANES:
        lane = "{}::{}".format(compiler, profile)
        for case in TIMESTRETCH_COHERENCE_PAIR_CASES:
            rows.append({
                "case": case,
                "class": "coherence-pair",
                "id": "timestretch::{}::coherence-pair::{}".format(
                    lane, case),
                "lane": lane,
                "oracle": "exact-classification-and-collision-tree-preservation",
            })
        for transition_id, _source, _target, _event \
                in TIMESTRETCH_COHERENCE_TRANSITIONS:
            rows.append({
                "class": "coherence-transition",
                "id": "timestretch::{}::coherence-transition::{}".format(
                    lane, transition_id),
                "lane": lane,
                "oracle": "exact-legal-successor-and-predecessor-binding",
                "transition_id": transition_id,
            })
        for field_case in TIMESTRETCH_COHERENCE_METADATA_CASES:
            rows.append({
                "class": "coherence-metadata",
                "field_case": field_case,
                "id": "timestretch::{}::coherence-metadata::{}".format(
                    lane, field_case),
                "lane": lane,
                "oracle": "named-divergence-rejected-with-unchanged-tree",
            })
    for mutant in TIMESTRETCH_COHERENCE_OUTER_MUTANTS:
        rows.append({
            "class": "coherence-outer-mutant",
            "id": "timestretch::coherence-outer-mutant::" + mutant,
            "lane": "outer-validator",
            "mutant": mutant,
            "oracle": "named-live-control-must-fail",
        })
    return tuple(rows)


EXPECTED_TIMESTRETCH_EXECUTION_IDS = expected_timestretch_execution_ids()
EXPECTED_TIMESTRETCH_EXECUTION_ROWS = expected_timestretch_execution_rows()
EXPECTED_TIMESTRETCH_EXECUTION_ROW_BY_ID = {
    row["id"]: row for row in EXPECTED_TIMESTRETCH_EXECUTION_ROWS
}
EXPECTED_TIMESTRETCH_EXECUTION_CARDINALITY = 2484
EXPECTED_THREADING_EXTERNAL_IDS = (
    "count-total",
    "count-template",
    "count-concrete",
    "count-overlap",
    "member-total-01",
    "member-total-02",
    "member-total-03",
    "member-total-04",
    "member-total-05",
    "member-total-06",
    "member-total-07",
    "member-total-08",
    "member-total-09",
    "member-total-10",
    "member-total-11",
    "member-template-01",
    "member-template-02",
    "member-template-03",
    "member-template-04",
    "member-template-05",
    "member-concrete-01",
    "member-concrete-02",
    "member-concrete-03",
    "member-concrete-04",
    "member-concrete-05",
    "member-concrete-06",
    "member-concrete-07",
    "member-concrete-08",
    "member-overlap-01",
    "member-overlap-02",
    "stale-10-5-7-2",
    "block-absent",
    "block-duplicate",
)
PACKAGE_R_ORACLE_LEGAL_POSITIVE_IDS = (
    "POS-CANONICAL",
    "POS-MULTILINE",
    "POS-REDUNDANT-PARENTHESES",
    "POS-COMMENTED-MULTILINE",
)
PACKAGE_R_ORACLE_META_CONTROL_IDS = (
    "PACKAGE-R-ORACLE-CALL-DELETE",
    "PACKAGE-R-ORACLE-CALL-NEUTRALIZE",
    "PACKAGE-R-ORACLE-CALL-DUPLICATE",
    "PACKAGE-R-ORACLE-CALL-WRONG-CALLEE",
    "PACKAGE-R-ORACLE-CALL-WRONG-ROOT",
    "PACKAGE-R-ORACLE-CALL-RESULT-DISCARD",
    "PACKAGE-R-ORACLE-CALL-WRONG-SINK",
    "PACKAGE-R-ORACLE-CALL-NESTED",
    "PACKAGE-R-ORACLE-CALL-AFTER-RETURN",
    "PACKAGE-R-ORACLE-INVARIANT-CALL-DELETE",
    "PACKAGE-R-ORACLE-INVARIANT-CALL-NEUTRALIZE",
    "PACKAGE-R-ORACLE-INVARIANT-HELPER-DELETE",
    "PACKAGE-R-ORACLE-INVARIANT-HELPER-RETURN-EMPTY",
    "PACKAGE-R-ORACLE-OUTER-PRODUCER-CHECK-DELETE",
    "PACKAGE-R-ORACLE-OUTER-PRODUCER-CHECK-NEUTRALIZE",
    "PACKAGE-R-ORACLE-OUTER-SELF-CHECK-DELETE",
    "PACKAGE-R-ORACLE-OUTER-CONTROLS-CALL-DELETE",
    "PACKAGE-R-ORACLE-OUTER-CONTROLS-CALL-NEUTRALIZE",
    "PACKAGE-R-ORACLE-OUTER-CONTROL-ID-DELETE",
    "PACKAGE-R-ORACLE-OUTER-CTEST-STRUCTURAL-CALL-DELETE",
    "PACKAGE-R-ORACLE-OUTER-CTEST-SELFTEST-CALL-DELETE",
    "PACKAGE-R-ORACLE-CONFIGURED-CTEST-BINDING-DELETE",
)
PACKAGE_R_ORACLE_EXPECTED_META_CONTROL_COUNT = 22
PACKAGE_R_ORACLE_META_CONTROL_INVENTORY_SHA256 = (
    "d999a44f0f1950f375f2422b19536161ce8fa27f130907baa739045b825b50d4"
)
LIVE_COMMAND = (
    "python3 -B tools/verify_global_validation_contract.py --live"
)
LIVE_PRODUCER_CALL = "        producer_result = run_live_producer(root, transcript_path)\n"
PRODUCER_MATRIX_CALL = "    transcript = build_live_transcript(root)\n"
PROCESS_GROUP_RACE_CONTROL_CALL = (
    "    controls.extend(process_group_members_race_control())\n"
)
LITERAL_CTEST_COMMAND = (
    "python3", "-B", "tools/verify_global_validation_contract.py", "--ctest",
)
REQUIRED_LITERAL_CTEST_REPETITIONS = 24
LITERAL_CTEST_TIMEOUT_SECONDS = 600.0
PROCESS_GROUP_RACE_TIMEOUT_SECONDS = 1.0
MAX_CAPTURE_BYTES = 1024 * 1024


def terminate_child(process: subprocess.Popen[bytes]) -> list[str]:
    actions: list[str] = []
    if os.name == "posix":
        for value, name in ((signal.SIGTERM, "SIGTERM_PROCESS_GROUP"),
                            (signal.SIGKILL, "SIGKILL_PROCESS_GROUP")):
            actions.append(name)
            try:
                os.killpg(process.pid, value)
            except ProcessLookupError:
                return actions
            try:
                process.wait(timeout=1)
                return actions
            except subprocess.TimeoutExpired:
                continue
    elif os.name == "nt":
        actions.append("TASKKILL_PROCESS_TREE")
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        actions.append("UNSUPPORTED_PROCESS_TREE_PLATFORM")
    return actions


def run_child(command: list[str], cwd: Path, timeout: float,
              environment: dict[str, str] | None = None) -> dict[str, object]:
    creationflags = 0
    keywords: dict[str, object] = {}
    if os.name == "posix":
        keywords["start_new_session"] = True
    elif os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        return {
            "command": command,
            "exit_code": None,
            "timed_out": False,
            "stdout": "",
            "stderr": "",
            "errors": ["UNSUPPORTED_PROCESS_TREE_PLATFORM"],
        }
    started = time.monotonic()
    process = subprocess.Popen(
        command, cwd=cwd, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        creationflags=creationflags, env=environment, **keywords)
    timed_out = False
    actions: list[str] = []
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        actions = terminate_child(process)
        stdout, stderr = process.communicate()
    stdout_large = len(stdout) > MAX_CAPTURE_BYTES
    stderr_large = len(stderr) > MAX_CAPTURE_BYTES
    stdout = stdout[:MAX_CAPTURE_BYTES]
    stderr = stderr[:MAX_CAPTURE_BYTES]
    errors: list[str] = []
    if timed_out:
        errors.append("CHILD_TIMEOUT")
    if "UNSUPPORTED_PROCESS_TREE_PLATFORM" in actions:
        errors.append("UNSUPPORTED_PROCESS_TREE_PLATFORM")
    if stdout_large:
        errors.append("STDOUT_LIMIT_EXCEEDED")
    if stderr_large:
        errors.append("STDERR_LIMIT_EXCEEDED")
    return {
        "command": command,
        "timeout_seconds": timeout,
        "duration_seconds": round(time.monotonic() - started, 6),
        "exit_code": process.returncode,
        "timed_out": timed_out,
        "stdout": stdout.decode("utf-8", "replace"),
        "stderr": stderr.decode("utf-8", "replace"),
        "cleanup_actions": actions,
        "errors": errors,
    }


def job_block(text: str, name: str) -> str:
    match = re.search(
        r"(?ms)^  {}:\n(.*?)(?=^  [A-Za-z0-9_-]+:\n|\Z)".format(
            re.escape(name)), text)
    return match.group(1) if match else ""


def binding_errors(root: Path,
                   overrides: dict[str, str] | None = None) -> list[str]:
    overrides = overrides or {}
    paths = (
        ".github/workflows/ci.yml",
        ".github/workflows/docs.yml",
        "tests/CMakeLists.txt",
    )
    text = {
        path: overrides.get(
            path, (root / path).read_text(encoding="ascii"))
        for path in paths
    }
    errors: list[str] = []
    ci_job = job_block(text[paths[0]], "documentation-quality")
    docs_job = job_block(text[paths[1]], "build-docs")
    for label, block in (("CI_DOCUMENTATION", ci_job),
                         ("DOCS_BUILD", docs_job)):
        if not block:
            errors.append("VALIDATION_OUTER_BINDING_JOB_MISSING:" + label)
            continue
        if "runs-on: ubuntu-24.04" not in block:
            errors.append("VALIDATION_OUTER_BINDING_RUNNER:" + label)
        if "g++-13" not in block or "clang-18" not in block:
            errors.append("VALIDATION_OUTER_BINDING_COMPILERS:" + label)
        if block.count(LIVE_COMMAND) != 1:
            errors.append("VALIDATION_OUTER_BINDING_LIVE:{}:{}".format(
                label, block.count(LIVE_COMMAND)))
    cmake = text[paths[2]]
    test_match = re.search(
        r"add_test\(NAME\s+global_product_corrections\s+"
        r"COMMAND\s+\$\{Python3_EXECUTABLE\}\s+-B\s+"
        r"\$\{PROJECT_SOURCE_DIR\}/tools/verify_global_validation_contract\.py\s+"
        r"--ctest\s+WORKING_DIRECTORY\s+\$\{PROJECT_SOURCE_DIR\}\s*\)",
        cmake, re.S)
    if test_match is None:
        errors.append("VALIDATION_OUTER_BINDING_CTEST")
    if len(re.findall(r"add_test\(NAME", cmake)) != 13:
        errors.append("VALIDATION_OUTER_CTEST_LOCAL_CARDINALITY")
    root_cmake = (root / "CMakeLists.txt").read_text(encoding="ascii")
    if len(re.findall(r"add_test\(NAME", root_cmake)) != 2:
        errors.append("VALIDATION_OUTER_CTEST_ROOT_CARDINALITY")
    return errors


def function_definition(tree: ast.AST, name: str) -> ast.FunctionDef | None:
    matches = [node for node in ast.walk(tree)
               if isinstance(node, ast.FunctionDef) and node.name == name]
    return matches[0] if len(matches) == 1 else None


def assignment_value(tree: ast.AST, name: str) -> ast.AST | None:
    matches: list[ast.AST] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == name
               for target in targets):
            matches.append(node.value)
    return matches[0] if len(matches) == 1 else None


def direct_call_count(node: ast.AST, name: str) -> int:
    return sum(
        isinstance(item, ast.Call)
        and isinstance(item.func, ast.Name)
        and item.func.id == name
        for item in ast.walk(node)
    )


def _rmo_top_level_function(
    tree: ast.Module, name: str,
) -> ast.FunctionDef | None:
    matches = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    return matches[0] if len(matches) == 1 else None


def _rmo_top_level_assignment(tree: ast.Module, name: str) -> ast.AST | None:
    matches: list[ast.AST] = []
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            targets = statement.targets
            value = statement.value
        elif isinstance(statement, ast.AnnAssign):
            targets = [statement.target]
            value = statement.value
        else:
            continue
        if any(isinstance(target, ast.Name) and target.id == name
               for target in targets):
            matches.append(value)
    return matches[0] if len(matches) == 1 else None


def _rmo_literal(tree: ast.Module, name: str) -> object | None:
    node = _rmo_top_level_assignment(tree, name)
    try:
        return ast.literal_eval(node) if node is not None else None
    except (TypeError, ValueError):
        return None


def _rmo_called_name(call: ast.Call, name: str) -> bool:
    return isinstance(call.func, ast.Name) and call.func.id == name


def _rmo_extend_argument(
    statement: ast.stmt, sink: str,
) -> ast.AST | None:
    if not isinstance(statement, ast.Expr) \
            or not isinstance(statement.value, ast.Call):
        return None
    call = statement.value
    if call.keywords or len(call.args) != 1:
        return None
    if not isinstance(call.func, ast.Attribute) or call.func.attr != "extend":
        return None
    if not isinstance(call.func.value, ast.Name) \
            or call.func.value.id != sink:
        return None
    return call.args[0]


def _rmo_exact_call_aggregation(
    statement: ast.stmt, sink: str, callee: str,
    arguments: tuple[str, ...],
) -> bool:
    value = _rmo_extend_argument(statement, sink)
    return (
        isinstance(value, ast.Call)
        and _rmo_called_name(value, callee)
        and not value.keywords
        and len(value.args) == len(arguments)
        and all(
            isinstance(argument, ast.Name) and argument.id == expected
            for argument, expected in zip(value.args, arguments)
        )
    )


def _rmo_direct_indexes(
    function: ast.FunctionDef, predicate: object,
) -> list[int]:
    return [
        index for index, statement in enumerate(function.body)
        if callable(predicate) and predicate(statement)
    ]


def _rmo_named_call_count(function: ast.FunctionDef, name: str) -> int:
    return sum(
        isinstance(node, ast.Call) and _rmo_called_name(node, name)
        for node in ast.walk(function)
    )


def _rmo_errors_initializations(
    function: ast.FunctionDef,
) -> tuple[list[int], list[int]]:
    assignments: list[int] = []
    empty_lists: list[int] = []
    for index, statement in enumerate(function.body):
        target: ast.AST | None = None
        value: ast.AST | None = None
        if isinstance(statement, ast.AnnAssign):
            target = statement.target
            value = statement.value
        elif isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            value = statement.value
        if isinstance(target, ast.Name) and target.id == "errors":
            assignments.append(index)
            if isinstance(value, ast.List) and not value.elts:
                empty_lists.append(index)
    return assignments, empty_lists


def _rmo_terminal_boundary(function: ast.FunctionDef) -> int | None:
    for index, statement in enumerate(function.body):
        if isinstance(statement, ast.For) \
                and isinstance(statement.iter, ast.Name) \
                and statement.iter.id == "errors":
            return index
        if isinstance(statement, ast.If) \
                and isinstance(statement.test, ast.Name) \
                and statement.test.id == "errors":
            return index
        if isinstance(statement, ast.Return):
            return index
    return None


def _rmo_producer_helper_errors(tree: ast.Module) -> list[str]:
    errors: list[str] = []
    checker = _rmo_top_level_function(tree, "package_oracle_binding_errors")
    current = _rmo_top_level_function(
        tree, "current_package_oracle_binding_errors")
    if checker is None or current is None:
        return ["PACKAGE_R_ORACLE_OUTER_PRODUCER_HELPER_MISSING"]
    checker_returns = [
        node for node in checker.body
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Call)
        and _rmo_called_name(node.value, "unique_errors")
        and len(node.value.args) == 1
        and isinstance(node.value.args[0], ast.Name)
        and node.value.args[0].id == "errors"
        and not node.value.keywords
    ]
    parse_calls = [
        node for node in ast.walk(checker)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "parse"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "ast"
    ]
    source_reads = [
        node for node in ast.walk(current)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "read_text"
        and isinstance(node.func.value, ast.Call)
        and isinstance(node.func.value.func, ast.Name)
        and node.func.value.func.id == "Path"
        and len(node.func.value.args) == 1
        and isinstance(node.func.value.args[0], ast.Name)
        and node.func.value.args[0].id == "__file__"
        and len(node.keywords) == 1
        and node.keywords[0].arg == "encoding"
        and isinstance(node.keywords[0].value, ast.Constant)
        and node.keywords[0].value.value == "ascii"
    ]
    if len(checker_returns) != 1 or len(parse_calls) != 1 \
            or _rmo_named_call_count(
                current, "package_oracle_binding_errors") != 1 \
            or len(source_reads) != 1:
        errors.append("PACKAGE_R_ORACLE_OUTER_PRODUCER_HELPER_NEUTRALIZED")
    return errors


def producer_package_oracle_binding_errors(
    producer_source: str,
) -> list[str]:
    """Independently audit the producer Package-R binding from its AST."""
    errors: list[str] = []
    try:
        tree = ast.parse(producer_source)
    except SyntaxError:
        return [
            "PACKAGE_R_ORACLE_OUTER_PRODUCER_CALL_DRIFT",
            "PACKAGE_R_ORACLE_OUTER_PRODUCER_POSITION",
        ]
    main_function = _rmo_top_level_function(tree, "main")
    if main_function is None:
        return list(dict.fromkeys([
            "PACKAGE_R_ORACLE_OUTER_PRODUCER_CALL_DRIFT",
            "PACKAGE_R_ORACLE_OUTER_PRODUCER_POSITION",
            *_rmo_producer_helper_errors(tree),
        ]))
    package_indexes = _rmo_direct_indexes(
        main_function,
        lambda statement: _rmo_exact_call_aggregation(
            statement, "errors", "package_errors", ("root",)),
    )
    invariant_indexes = _rmo_direct_indexes(
        main_function,
        lambda statement: _rmo_exact_call_aggregation(
            statement, "errors", "current_package_oracle_binding_errors", ()),
    )
    package_calls = _rmo_named_call_count(main_function, "package_errors")
    invariant_calls = _rmo_named_call_count(
        main_function, "current_package_oracle_binding_errors")
    assignments, empty_initializations = \
        _rmo_errors_initializations(main_function)
    boundary = _rmo_terminal_boundary(main_function)
    if package_calls == 0:
        errors.append("PACKAGE_R_ORACLE_OUTER_PRODUCER_MISSING")
    if package_calls > 1 or len(package_indexes) > 1:
        errors.append("PACKAGE_R_ORACLE_OUTER_PRODUCER_DUPLICATE")
    if package_calls != 1 or len(package_indexes) != 1:
        errors.append("PACKAGE_R_ORACLE_OUTER_PRODUCER_CALL_DRIFT")
    if invariant_calls != 1 or len(invariant_indexes) != 1:
        errors.append("PACKAGE_R_ORACLE_OUTER_PRODUCER_INVARIANT_MISSING")
    if (
        len(assignments) != 1
        or len(empty_initializations) != 1
        or len(package_indexes) != 1
        or len(invariant_indexes) != 1
        or boundary is None
        or not (
            empty_initializations[0]
            < invariant_indexes[0]
            < package_indexes[0]
            < boundary
        )
    ):
        errors.append("PACKAGE_R_ORACLE_OUTER_PRODUCER_POSITION")
    errors.extend(_rmo_producer_helper_errors(tree))
    return list(dict.fromkeys(errors))


def _rmo_result_append_indexes(function: ast.FunctionDef) -> list[int]:
    indexes: list[int] = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call) or node.keywords \
                or len(node.args) != 1:
            continue
        if not isinstance(node.func, ast.Attribute) \
                or node.func.attr != "append" \
                or not isinstance(node.func.value, ast.Name) \
                or node.func.value.id != "results":
            continue
        value = node.args[0]
        if not isinstance(value, ast.Tuple) or len(value.elts) != 2:
            continue
        identity = value.elts[0]
        if not isinstance(identity, ast.Subscript) \
                or not isinstance(identity.value, ast.Name) \
                or identity.value.id not in (
                    "PACKAGE_R_ORACLE_META_CONTROL_IDS", "control_ids") \
                or not isinstance(identity.slice, ast.Constant) \
                or not isinstance(identity.slice.value, int):
            continue
        indexes.append(identity.slice.value)
    return indexes



def _rmo_outer_bootstrap_surface_errors(tree: ast.Module) -> list[str]:
    errors: list[str] = []
    bootstrap = _rmo_top_level_function(
        tree, "package_r_outer_executable_bootstrap_errors")
    if bootstrap is None:
        return ["PACKAGE_R_ORACLE_OUTER_BOOTSTRAP_HELPER_MISSING"]
    constants = {
        node.value for node in ast.walk(bootstrap)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    required = {
        "PACKAGE_R_ORACLE_OUTER_SOURCE_SYNTAX",
        "PACKAGE_R_ORACLE_OUTER_PRODUCER_HELPER_MISSING",
        "PACKAGE_R_ORACLE_OUTER_PRODUCER_BOOTSTRAP_BOUNDARY",
        "PACKAGE_R_ORACLE_OUTER_PRODUCER_BOOTSTRAP_NEUTRALIZED",
        "PACKAGE_R_ORACLE_OUTER_CHECKER_MISSING",
        "PACKAGE_R_ORACLE_OUTER_CONTROL_HELPER",
        "PACKAGE_R_ORACLE_OUTER_CTEST_STRUCTURAL_CALL",
    }
    parse_calls = [
        node for node in ast.walk(bootstrap)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "parse"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "ast"
    ]
    if [item.arg for item in bootstrap.args.args] != [
            "source", "producer_source"] or len(parse_calls) != 2 \
            or not required.issubset(constants) \
            or not any(isinstance(node, ast.Return)
                       and isinstance(node.value, ast.Call)
                       and isinstance(node.value.func, ast.Name)
                       and node.value.func.id == "list"
                       for node in ast.walk(bootstrap)):
        errors.append("PACKAGE_R_ORACLE_OUTER_BOOTSTRAP_NEUTRALIZED")
    main_function = _rmo_top_level_function(tree, "main")
    if main_function is None:
        return errors + ["PACKAGE_R_ORACLE_OUTER_BOOTSTRAP_BOUNDARY"]
    get_calls = [
        node for node in ast.walk(main_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Call)
        and isinstance(node.func.value.func, ast.Name)
        and node.func.value.func.id == "globals"
        and node.func.attr == "get"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "package_r_outer_executable_bootstrap_errors"
    ]
    bootstrap_calls = [
        node for node in ast.walk(main_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "package_r_outer_bootstrap"
    ]
    caught: list[str] = []
    try_nodes = [
        node for node in ast.walk(main_function)
        if isinstance(node, ast.Try)
        and any(isinstance(item, ast.Call)
                and isinstance(item.func, ast.Name)
                and item.func.id == "package_r_outer_bootstrap"
                for item in ast.walk(node))
    ]
    for node in try_nodes:
        for handler in node.handlers:
            if isinstance(handler.type, ast.Tuple):
                caught.extend(item.id for item in handler.type.elts
                              if isinstance(item, ast.Name))
    expected = [
        "AttributeError", "NameError", "OSError", "SyntaxError",
        "TypeError", "UnicodeError", "ValueError",
    ]
    if len(get_calls) != 1 or len(bootstrap_calls) != 1 \
            or len(try_nodes) != 1 \
            or sorted(caught) != sorted(expected):
        errors.append("PACKAGE_R_ORACLE_OUTER_BOOTSTRAP_BOUNDARY")
    return errors


def package_oracle_outer_binding_errors(source: str) -> list[str]:
    """Audit the independent outer surface and its mutual bindings."""
    errors: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ["PACKAGE_R_ORACLE_OUTER_SOURCE_SYNTAX"]
    errors.extend(_rmo_outer_bootstrap_surface_errors(tree))
    inventory = _rmo_literal(tree, "PACKAGE_R_ORACLE_META_CONTROL_IDS")
    expected_count = _rmo_literal(
        tree, "PACKAGE_R_ORACLE_EXPECTED_META_CONTROL_COUNT")
    inventory_digest = _rmo_literal(
        tree, "PACKAGE_R_ORACLE_META_CONTROL_INVENTORY_SHA256")
    observed_digest = None
    if isinstance(inventory, tuple) \
            and all(isinstance(item, str) for item in inventory):
        observed_digest = hashlib.sha256(
            ("\n".join(inventory) + "\n").encode("ascii")
        ).hexdigest()
    if (
        not isinstance(inventory, tuple)
        or len(inventory) != 22
        or len(set(inventory)) != 22
        or expected_count != 22
        or inventory_digest
            != "d999a44f0f1950f375f2422b19536161ce8fa27f130907baa739045b825b50d4"
        or observed_digest
            != "d999a44f0f1950f375f2422b19536161ce8fa27f130907baa739045b825b50d4"
    ):
        errors.append("PACKAGE_R_ORACLE_OUTER_CONTROL_INVENTORY")
    positive_inventory = _rmo_literal(
        tree, "PACKAGE_R_ORACLE_LEGAL_POSITIVE_IDS")
    if positive_inventory != (
        "POS-CANONICAL", "POS-MULTILINE", "POS-REDUNDANT-PARENTHESES",
        "POS-COMMENTED-MULTILINE",
    ):
        errors.append("PACKAGE_R_ORACLE_OUTER_POSITIVE_INVENTORY")

    structural = _rmo_top_level_function(tree, "structural_errors")
    if structural is None:
        errors.append("PACKAGE_R_ORACLE_OUTER_STRUCTURAL_FUNCTION_MISSING")
    else:
        producer_checks = _rmo_direct_indexes(
            structural,
            lambda statement: _rmo_exact_call_aggregation(
                statement, "errors",
                "producer_package_oracle_binding_errors",
                ("producer_source",)),
        )
        self_checks = _rmo_direct_indexes(
            structural,
            lambda statement: _rmo_exact_call_aggregation(
                statement, "errors", "package_oracle_outer_binding_errors",
                ("source",)),
        )
        if len(producer_checks) != 1 or _rmo_named_call_count(
                structural, "producer_package_oracle_binding_errors") != 1:
            errors.append("PACKAGE_R_ORACLE_OUTER_CHECK_MISSING")
        if len(self_checks) != 1 or _rmo_named_call_count(
                structural, "package_oracle_outer_binding_errors") != 1:
            errors.append("PACKAGE_R_ORACLE_OUTER_SELF_CHECK_MISSING")

    self_test_function = _rmo_top_level_function(tree, "self_test")
    if self_test_function is None:
        errors.append("PACKAGE_R_ORACLE_OUTER_SELFTEST_MISSING")
    else:
        controls_calls = _rmo_direct_indexes(
            self_test_function,
            lambda statement: _rmo_exact_call_aggregation(
                statement, "controls", "package_oracle_meta_control_results",
                ("source", "producer_source")),
        )
        positive_calls = _rmo_direct_indexes(
            self_test_function,
            lambda statement: _rmo_exact_call_aggregation(
                statement, "controls",
                "package_oracle_legal_positive_results",
                ("producer_source",)),
        )
        if len(controls_calls) != 1 or _rmo_named_call_count(
                self_test_function,
                "package_oracle_meta_control_results") != 1:
            errors.append("PACKAGE_R_ORACLE_OUTER_CONTROLS_CALL_MISSING")
        if len(positive_calls) != 1 or _rmo_named_call_count(
                self_test_function,
                "package_oracle_legal_positive_results") != 1:
            errors.append("PACKAGE_R_ORACLE_OUTER_POSITIVES_CALL_MISSING")

    ctest_function = _rmo_top_level_function(tree, "ctest_mode")
    if ctest_function is None:
        errors.append("PACKAGE_R_ORACLE_OUTER_CTEST_MODE_MISSING")
    else:
        if _rmo_named_call_count(ctest_function, "structural_errors") != 1:
            errors.append("PACKAGE_R_ORACLE_OUTER_CTEST_STRUCTURAL_CALL")
        if _rmo_named_call_count(ctest_function, "self_test") != 1:
            errors.append("PACKAGE_R_ORACLE_OUTER_CTEST_SELFTEST_CALL")
        if _rmo_named_call_count(ctest_function, "normal_gate") != 1:
            errors.append("PACKAGE_R_ORACLE_OUTER_CTEST_NORMAL_GATE_CALL")

    controls_function = _rmo_top_level_function(
        tree, "package_oracle_meta_control_results")
    if controls_function is None \
            or [argument.arg for argument in controls_function.args.args] \
            != ["source", "producer_source"] \
            or _rmo_result_append_indexes(controls_function) != list(range(22)) \
            or len([
                node for node in controls_function.body
                if isinstance(node, ast.Return)
                and isinstance(node.value, ast.Name)
                and node.value.id == "results"
            ]) != 1:
        errors.append("PACKAGE_R_ORACLE_OUTER_CONTROL_HELPER")
    positives_function = _rmo_top_level_function(
        tree, "package_oracle_legal_positive_results")
    if positives_function is None \
            or [argument.arg for argument in positives_function.args.args] \
            != ["producer_source"] \
            or _rmo_named_call_count(
                positives_function, "producer_package_oracle_binding_errors") \
            != 4:
        errors.append("PACKAGE_R_ORACLE_OUTER_POSITIVE_HELPER")
    return list(dict.fromkeys(errors))


def _rmo_replace_once(source: str, old: str, new: str) -> str | None:
    if source.count(old) != 1:
        return None
    return source.replace(old, new, 1)


def _rmo_delete_producer_checker(producer_source: str) -> str | None:
    start = producer_source.find("def package_oracle_binding_errors(")
    finish = producer_source.find(
        "def current_package_oracle_binding_errors(", start + 1)
    if start < 0 or finish < 0:
        return None
    return producer_source[:start] + producer_source[finish:]


def _rmo_canonicalize_package_call(producer_source: str) -> str | None:
    try:
        tree = ast.parse(producer_source)
    except SyntaxError:
        return None
    main_function = _rmo_top_level_function(tree, "main")
    if main_function is None:
        return None
    statements = [
        statement for statement in main_function.body
        if _rmo_exact_call_aggregation(
            statement, "errors", "package_errors", ("root",))
    ]
    if len(statements) != 1:
        return None
    statement = statements[0]
    if statement.end_lineno is None:
        return None
    lines = producer_source.splitlines(keepends=True)
    lines[statement.lineno - 1:statement.end_lineno] = [
        "    errors.extend(package_errors(root))\n"]
    return "".join(lines)


def package_oracle_legal_positive_results(
    producer_source: str,
) -> list[tuple[str, bool]]:
    """Exercise the four legal AST-equivalent Package-R call forms."""
    normalized = _rmo_canonicalize_package_call(producer_source)
    if normalized is None:
        return [
            (identity, False)
            for identity in PACKAGE_R_ORACLE_LEGAL_POSITIVE_IDS
        ]
    producer_source = normalized
    canonical = "    errors.extend(package_errors(root))\n"
    multiline = _rmo_replace_once(
        producer_source, canonical,
        "    errors.extend(\n        package_errors(root)\n    )\n")
    redundant = _rmo_replace_once(
        producer_source, canonical,
        "    errors.extend(((package_errors((root)))))\n")
    commented = _rmo_replace_once(
        producer_source, canonical,
        "    errors.extend(  # required aggregation\n"
        "        package_errors(  # exact root\n"
        "            root\n"
        "        )\n"
        "    )\n")
    return [
        (PACKAGE_R_ORACLE_LEGAL_POSITIVE_IDS[0],
         not producer_package_oracle_binding_errors(producer_source)),
        (PACKAGE_R_ORACLE_LEGAL_POSITIVE_IDS[1], multiline is not None
         and not producer_package_oracle_binding_errors(multiline)),
        (PACKAGE_R_ORACLE_LEGAL_POSITIVE_IDS[2], redundant is not None
         and not producer_package_oracle_binding_errors(redundant)),
        (PACKAGE_R_ORACLE_LEGAL_POSITIVE_IDS[3], commented is not None
         and not producer_package_oracle_binding_errors(commented)),
    ]


def package_oracle_meta_control_results(
    source: str, producer_source: str,
) -> list[tuple[str, bool]]:
    """Exercise the literal ordered 22-control meta-oracle inventory."""
    results: list[tuple[str, bool]] = []
    control_ids = (
        PACKAGE_R_ORACLE_META_CONTROL_IDS
        + tuple(
            "PACKAGE-R-ORACLE-MISSING-CONTROL-{:02d}".format(index)
            for index in range(22)
        )
    )[:22]
    canonical_source = _rmo_canonicalize_package_call(producer_source)
    if canonical_source is not None:
        producer_source = canonical_source
    package_call = "    errors.extend(package_errors(root))\n"
    invariant_call = (
        "    errors.extend(current_package_oracle_binding_errors())\n")
    outer_producer_check = (
        "    errors.extend("
        "producer_package_oracle_binding_errors(producer_source))\n")
    outer_self_check = (
        "    errors.extend(package_oracle_outer_binding_errors(source))\n")
    outer_controls_call = (
        "    controls.extend("
        "package_oracle_meta_control_results(source, producer_source))\n")

    changed = _rmo_replace_once(producer_source, package_call, "")
    results.append((control_ids[0],
                    changed is not None and
                    "PACKAGE_R_ORACLE_OUTER_PRODUCER_MISSING"
                    in producer_package_oracle_binding_errors(changed)))
    changed = _rmo_replace_once(
        producer_source, package_call, "    errors.extend([])\n")
    results.append((control_ids[1],
                    changed is not None and
                    "PACKAGE_R_ORACLE_OUTER_PRODUCER_MISSING"
                    in producer_package_oracle_binding_errors(changed)))
    changed = _rmo_replace_once(
        producer_source, package_call, package_call + package_call)
    results.append((control_ids[2],
                    changed is not None and
                    "PACKAGE_R_ORACLE_OUTER_PRODUCER_DUPLICATE"
                    in producer_package_oracle_binding_errors(changed)))
    changed = _rmo_replace_once(
        producer_source, package_call,
        "    errors.extend(package_mutant_control_errors(root))\n")
    results.append((control_ids[3],
                    changed is not None and
                    "PACKAGE_R_ORACLE_OUTER_PRODUCER_MISSING"
                    in producer_package_oracle_binding_errors(changed)))
    changed = _rmo_replace_once(
        producer_source, package_call,
        "    errors.extend(package_errors(ROOT))\n")
    results.append((control_ids[4],
                    changed is not None and
                    "PACKAGE_R_ORACLE_OUTER_PRODUCER_CALL_DRIFT"
                    in producer_package_oracle_binding_errors(changed)))
    changed = _rmo_replace_once(
        producer_source, package_call, "    package_errors(root)\n")
    results.append((control_ids[5],
                    changed is not None and
                    "PACKAGE_R_ORACLE_OUTER_PRODUCER_CALL_DRIFT"
                    in producer_package_oracle_binding_errors(changed)))
    changed = _rmo_replace_once(
        producer_source, package_call,
        "    other_errors.extend(package_errors(root))\n")
    results.append((control_ids[6],
                    changed is not None and
                    "PACKAGE_R_ORACLE_OUTER_PRODUCER_CALL_DRIFT"
                    in producer_package_oracle_binding_errors(changed)))
    changed = _rmo_replace_once(
        producer_source, package_call,
        "    if not arguments.skip_public_text:\n"
        "        errors.extend(package_errors(root))\n")
    results.append((control_ids[7],
                    changed is not None and
                    "PACKAGE_R_ORACLE_OUTER_PRODUCER_CALL_DRIFT"
                    in producer_package_oracle_binding_errors(changed)))
    moved = _rmo_replace_once(producer_source, package_call, "")
    changed = None if moved is None else _rmo_replace_once(
        moved, "    return 0\n\n\nif __name__ == \"__main__\":\n",
        "    return 0\n    errors.extend(package_errors(root))\n\n\n"
        "if __name__ == \"__main__\":\n")
    results.append((control_ids[8],
                    changed is not None and
                    "PACKAGE_R_ORACLE_OUTER_PRODUCER_POSITION"
                    in producer_package_oracle_binding_errors(changed)))
    changed = _rmo_replace_once(producer_source, invariant_call, "")
    results.append((control_ids[9],
                    changed is not None and
                    "PACKAGE_R_ORACLE_OUTER_PRODUCER_INVARIANT_MISSING"
                    in producer_package_oracle_binding_errors(changed)))
    changed = _rmo_replace_once(
        producer_source, invariant_call, "    errors.extend([])\n")
    results.append((control_ids[10],
                    changed is not None and
                    "PACKAGE_R_ORACLE_OUTER_PRODUCER_INVARIANT_MISSING"
                    in producer_package_oracle_binding_errors(changed)))
    changed = _rmo_delete_producer_checker(producer_source)
    results.append((control_ids[11],
                    changed is not None and
                    "PACKAGE_R_ORACLE_OUTER_PRODUCER_HELPER_MISSING"
                    in producer_package_oracle_binding_errors(changed)))
    changed = _rmo_replace_once(
        producer_source,
        "    return unique_errors(errors)\n\n\n"
        "def current_package_oracle_binding_errors",
        "    return []\n\n\n"
        "def current_package_oracle_binding_errors")
    results.append((control_ids[12],
                    changed is not None and
                    "PACKAGE_R_ORACLE_OUTER_PRODUCER_HELPER_NEUTRALIZED"
                    in producer_package_oracle_binding_errors(changed)))
    changed = _rmo_replace_once(source, outer_producer_check, "")
    results.append((control_ids[13],
                    changed is not None and
                    "PACKAGE_R_ORACLE_OUTER_CHECK_MISSING"
                    in package_oracle_outer_binding_errors(changed)))
    changed = _rmo_replace_once(
        source, outer_producer_check, "    errors.extend([])\n")
    results.append((control_ids[14],
                    changed is not None and
                    "PACKAGE_R_ORACLE_OUTER_CHECK_MISSING"
                    in package_oracle_outer_binding_errors(changed)))
    changed = _rmo_replace_once(source, outer_self_check, "")
    results.append((control_ids[15],
                    changed is not None and
                    "PACKAGE_R_ORACLE_OUTER_SELF_CHECK_MISSING"
                    in package_oracle_outer_binding_errors(changed)))
    changed = _rmo_replace_once(source, outer_controls_call, "")
    results.append((control_ids[16],
                    changed is not None and
                    "PACKAGE_R_ORACLE_OUTER_CONTROLS_CALL_MISSING"
                    in package_oracle_outer_binding_errors(changed)))
    changed = _rmo_replace_once(
        source, outer_controls_call, "    controls.extend([])\n")
    results.append((control_ids[17],
                    changed is not None and
                    "PACKAGE_R_ORACLE_OUTER_CONTROLS_CALL_MISSING"
                    in package_oracle_outer_binding_errors(changed)))
    changed = _rmo_replace_once(
        source, '    "PACKAGE-R-ORACLE-CALL-DELETE",\n', "")
    results.append((control_ids[18],
                    changed is not None and
                    "PACKAGE_R_ORACLE_OUTER_CONTROL_INVENTORY"
                    in package_oracle_outer_binding_errors(changed)))
    changed = _rmo_replace_once(
        source,
        '    errors = structural_errors('
        'Path(__file__).read_text(encoding="ascii"))\n',
        "    errors: list[str] = []\n")
    results.append((control_ids[19],
                    changed is not None and
                    "PACKAGE_R_ORACLE_OUTER_CTEST_STRUCTURAL_CALL"
                    in package_oracle_outer_binding_errors(changed)))
    changed = _rmo_replace_once(
        source, "    if self_test(root) != 0:\n", "    if False:\n")
    results.append((control_ids[20],
                    changed is not None and
                    "PACKAGE_R_ORACLE_OUTER_CTEST_SELFTEST_CALL"
                    in package_oracle_outer_binding_errors(changed)))
    cmake_path = ROOT / "tests/CMakeLists.txt"
    try:
        cmake_source = cmake_path.read_text(encoding="ascii")
    except (OSError, UnicodeError):
        cmake_mutant = None
    else:
        cmake_mutant = _rmo_replace_once(
            cmake_source,
            "${PROJECT_SOURCE_DIR}/tools/verify_global_validation_contract.py",
            "${PROJECT_SOURCE_DIR}/tools/verify_m018_global_corrections.py")
    binding_result = [] if cmake_mutant is None else binding_errors(
        ROOT, {"tests/CMakeLists.txt": cmake_mutant})
    results.append((control_ids[21],
                    "VALIDATION_OUTER_BINDING_CTEST" in binding_result))
    return results



def package_r_outer_executable_bootstrap_errors(
    source: str, producer_source: str,
) -> list[str]:
    """Audit both executable surfaces without using either shared matcher."""
    errors: list[str] = []
    try:
        outer_tree = ast.parse(source)
    except SyntaxError:
        return ["PACKAGE_R_ORACLE_OUTER_SOURCE_SYNTAX"]
    try:
        producer_tree = ast.parse(producer_source)
    except SyntaxError:
        return [
            "PACKAGE_R_ORACLE_OUTER_PRODUCER_CALL_DRIFT",
            "PACKAGE_R_ORACLE_OUTER_PRODUCER_POSITION",
        ]

    def top(tree: ast.Module, name: str) -> ast.FunctionDef | None:
        matches = [
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        ]
        return matches[0] if len(matches) == 1 else None

    def called(call: ast.Call, name: str) -> bool:
        return isinstance(call.func, ast.Name) and call.func.id == name

    def named_count(function: ast.FunctionDef, name: str) -> int:
        return sum(isinstance(node, ast.Call) and called(node, name)
                   for node in ast.walk(function))

    def exact_extend(statement: ast.stmt, sink: str, callee: str,
                     arguments: tuple[str, ...]) -> bool:
        if not isinstance(statement, ast.Expr) \
                or not isinstance(statement.value, ast.Call):
            return False
        outer_call = statement.value
        if outer_call.keywords or len(outer_call.args) != 1 \
                or not isinstance(outer_call.func, ast.Attribute) \
                or outer_call.func.attr != "extend" \
                or not isinstance(outer_call.func.value, ast.Name) \
                or outer_call.func.value.id != sink:
            return False
        value = outer_call.args[0]
        return isinstance(value, ast.Call) and called(value, callee) \
            and not value.keywords and len(value.args) == len(arguments) \
            and all(isinstance(argument, ast.Name)
                    and argument.id == expected
                    for argument, expected in zip(value.args, arguments))

    producer_main = top(producer_tree, "main")
    producer_checker = top(producer_tree, "package_oracle_binding_errors")
    producer_current = top(
        producer_tree, "current_package_oracle_binding_errors")
    producer_bootstrap = top(
        producer_tree, "package_r_executable_bootstrap_errors")
    if producer_checker is None or producer_current is None \
            or producer_bootstrap is None:
        errors.append("PACKAGE_R_ORACLE_OUTER_PRODUCER_HELPER_MISSING")
    elif [item.arg for item in producer_checker.args.args] != ["source"] \
            or producer_current.args.args \
            or [item.arg for item in producer_bootstrap.args.args] != ["source"]:
        errors.append("PACKAGE_R_ORACLE_OUTER_PRODUCER_HELPER_NEUTRALIZED")
    if producer_bootstrap is not None:
        producer_bootstrap_constants = {
            node.value for node in ast.walk(producer_bootstrap)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        producer_bootstrap_required = {
            "PACKAGE_R_ORACLE_PRODUCER_CALL_DRIFT",
            "PACKAGE_R_ORACLE_PRODUCER_POSITION",
            "PACKAGE_R_ORACLE_PRODUCER_HELPER_MISSING",
            "PACKAGE_R_ORACLE_PRODUCER_HELPER_NEUTRALIZED",
            "PACKAGE_R_ORACLE_PRODUCER_INVARIANT_MISSING",
        }
        producer_bootstrap_parse_calls = [
            node for node in ast.walk(producer_bootstrap)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "ast" and node.func.attr == "parse"
        ]
        producer_bootstrap_returns = [
            node for node in ast.walk(producer_bootstrap)
            if isinstance(node, ast.Return)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "list"
        ]
        if len(producer_bootstrap_parse_calls) != 1 \
                or not producer_bootstrap_returns \
                or not producer_bootstrap_required.issubset(
                    producer_bootstrap_constants):
            errors.append(
                "PACKAGE_R_ORACLE_OUTER_PRODUCER_BOOTSTRAP_NEUTRALIZED")
    if producer_checker is not None and producer_current is not None:
        producer_checker_parse_calls = [
            node for node in ast.walk(producer_checker)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "ast" and node.func.attr == "parse"
        ]
        producer_checker_returns = [
            node for node in ast.walk(producer_checker)
            if isinstance(node, ast.Return)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "unique_errors"
        ]
        producer_current_calls = named_count(
            producer_current, "package_oracle_binding_errors")
        producer_current_reads = [
            node for node in ast.walk(producer_current)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "read_text"
            and any(isinstance(item, ast.Name) and item.id == "__file__"
                    for item in ast.walk(node.func.value))
        ]
        if len(producer_checker_parse_calls) != 1 \
                or len(producer_checker_returns) < 1 \
                or producer_current_calls != 1 \
                or len(producer_current_reads) != 1:
            errors.append("PACKAGE_R_ORACLE_OUTER_PRODUCER_HELPER_NEUTRALIZED")
    if producer_main is None:
        errors.extend((
            "PACKAGE_R_ORACLE_OUTER_PRODUCER_CALL_DRIFT",
            "PACKAGE_R_ORACLE_OUTER_PRODUCER_POSITION",
        ))
    else:
        package_indexes = [
            index for index, statement in enumerate(producer_main.body)
            if exact_extend(statement, "errors", "package_errors", ("root",))
        ]
        invariant_indexes = [
            index for index, statement in enumerate(producer_main.body)
            if exact_extend(statement, "errors",
                            "current_package_oracle_binding_errors", ())
        ]
        package_calls = named_count(producer_main, "package_errors")
        invariant_calls = named_count(
            producer_main, "current_package_oracle_binding_errors")
        if package_calls == 0:
            errors.append("PACKAGE_R_ORACLE_OUTER_PRODUCER_MISSING")
        if package_calls > 1 or len(package_indexes) > 1:
            errors.append("PACKAGE_R_ORACLE_OUTER_PRODUCER_DUPLICATE")
        if package_calls != 1 or len(package_indexes) != 1:
            errors.append("PACKAGE_R_ORACLE_OUTER_PRODUCER_CALL_DRIFT")
        if invariant_calls != 1 or len(invariant_indexes) != 1:
            errors.append("PACKAGE_R_ORACLE_OUTER_PRODUCER_INVARIANT_MISSING")
        if len(package_indexes) != 1 or len(invariant_indexes) != 1 \
                or invariant_indexes[0] >= package_indexes[0]:
            errors.append("PACKAGE_R_ORACLE_OUTER_PRODUCER_POSITION")
        producer_bootstrap_get_calls = [
            node for node in ast.walk(producer_main)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Call)
            and isinstance(node.func.value.func, ast.Name)
            and node.func.value.func.id == "globals"
            and node.func.attr == "get"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == (
                "package_r_executable_bootstrap_errors")
        ]
        producer_bootstrap_calls = [
            node for node in ast.walk(producer_main)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "package_r_bootstrap"
        ]
        producer_bootstrap_reads = [
            node for node in ast.walk(producer_main)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "read_text"
            and any(isinstance(item, ast.Name) and item.id == "__file__"
                    for item in ast.walk(node.func.value))
        ]
        producer_bootstrap_try_nodes = [
            node for node in ast.walk(producer_main)
            if isinstance(node, ast.Try)
            and any(isinstance(item, ast.Call)
                    and isinstance(item.func, ast.Name)
                    and item.func.id == "package_r_bootstrap"
                    for item in ast.walk(node))
        ]
        producer_bootstrap_caught: list[str] = []
        for node in producer_bootstrap_try_nodes:
            for handler in node.handlers:
                if isinstance(handler.type, ast.Tuple):
                    producer_bootstrap_caught.extend(
                        item.id for item in handler.type.elts
                        if isinstance(item, ast.Name))
        producer_bootstrap_expected = [
            "AttributeError", "NameError", "OSError", "SyntaxError",
            "TypeError", "UnicodeError", "ValueError",
        ]
        if len(producer_bootstrap_get_calls) != 1 \
                or len(producer_bootstrap_calls) != 1 \
                or len(producer_bootstrap_reads) < 1 \
                or len(producer_bootstrap_try_nodes) != 1 \
                or sorted(producer_bootstrap_caught) != sorted(
                    producer_bootstrap_expected):
            errors.append("PACKAGE_R_ORACLE_OUTER_PRODUCER_BOOTSTRAP_BOUNDARY")

    inventory_nodes: list[ast.AST] = []
    for statement in outer_tree.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) \
            else [statement.target]
        if any(isinstance(target, ast.Name)
               and target.id == "PACKAGE_R_ORACLE_META_CONTROL_IDS"
               for target in targets):
            inventory_nodes.append(statement.value)
    try:
        inventory = ast.literal_eval(inventory_nodes[0]) \
            if len(inventory_nodes) == 1 else None
    except (TypeError, ValueError):
        inventory = None
    inventory_digest = hashlib.sha256(
        (("\n".join(inventory) + "\n") if isinstance(inventory, tuple)
         and all(isinstance(item, str) for item in inventory) else "")
        .encode("ascii")).hexdigest()
    if not isinstance(inventory, tuple) or len(inventory) != 22 \
            or len(set(inventory)) != 22 \
            or inventory_digest != (
                "d999a44f0f1950f375f2422b19536161ce8fa27f130907baa739045b825b50d4"):
        errors.append("PACKAGE_R_ORACLE_OUTER_CONTROL_INVENTORY")

    structural = top(outer_tree, "structural_errors")
    outer_checker = top(outer_tree, "package_oracle_outer_binding_errors")
    producer_checker_outer = top(
        outer_tree, "producer_package_oracle_binding_errors")
    controls = top(outer_tree, "package_oracle_meta_control_results")
    self_test_function = top(outer_tree, "self_test")
    ctest = top(outer_tree, "ctest_mode")
    if producer_checker_outer is None:
        errors.append("PACKAGE_R_ORACLE_OUTER_PRODUCER_HELPER_MISSING")
    if outer_checker is None:
        errors.append("PACKAGE_R_ORACLE_OUTER_CHECKER_MISSING")
    if controls is None:
        errors.append("PACKAGE_R_ORACLE_OUTER_CONTROL_HELPER")
    if structural is None:
        errors.append("PACKAGE_R_ORACLE_OUTER_STRUCTURAL_FUNCTION_MISSING")
    else:
        producer_checks = [
            statement for statement in structural.body
            if exact_extend(statement, "errors",
                            "producer_package_oracle_binding_errors",
                            ("producer_source",))
        ]
        self_checks = [
            statement for statement in structural.body
            if exact_extend(statement, "errors",
                            "package_oracle_outer_binding_errors", ("source",))
        ]
        if len(producer_checks) != 1 \
                or named_count(structural,
                               "producer_package_oracle_binding_errors") != 1:
            errors.append("PACKAGE_R_ORACLE_OUTER_CHECK_MISSING")
        if len(self_checks) != 1 \
                or named_count(structural,
                               "package_oracle_outer_binding_errors") != 1:
            errors.append("PACKAGE_R_ORACLE_OUTER_SELF_CHECK_MISSING")
    if self_test_function is None:
        errors.append("PACKAGE_R_ORACLE_OUTER_SELFTEST_MISSING")
    else:
        control_calls = [
            statement for statement in self_test_function.body
            if exact_extend(statement, "controls",
                            "package_oracle_meta_control_results",
                            ("source", "producer_source"))
        ]
        if len(control_calls) != 1 \
                or named_count(self_test_function,
                               "package_oracle_meta_control_results") != 1:
            errors.append("PACKAGE_R_ORACLE_OUTER_CONTROLS_CALL_MISSING")
    if ctest is None:
        errors.append("PACKAGE_R_ORACLE_OUTER_CTEST_MODE_MISSING")
    else:
        if named_count(ctest, "structural_errors") != 1:
            errors.append("PACKAGE_R_ORACLE_OUTER_CTEST_STRUCTURAL_CALL")
        if named_count(ctest, "self_test") != 1:
            errors.append("PACKAGE_R_ORACLE_OUTER_CTEST_SELFTEST_CALL")
        if named_count(ctest, "normal_gate") != 1:
            errors.append("PACKAGE_R_ORACLE_OUTER_CTEST_NORMAL_GATE_CALL")
    return list(dict.fromkeys(errors))


def process_group_race_structure_errors(
    source: str, producer_source: str,
) -> list[str]:
    errors: list[str] = []
    try:
        tree = ast.parse(source)
        producer_tree = ast.parse(producer_source)
    except SyntaxError as error:
        return ["PROCESS_GROUP_RACE_SOURCE_SYNTAX:" + str(error)]

    control = function_definition(tree, "process_group_members_race_control")
    self_test_function = function_definition(tree, "self_test")
    ctest_function = function_definition(tree, "ctest_mode")
    repetition = function_definition(tree, "literal_ctest_repetition_errors")
    if control is None:
        errors.append("PROCESS_GROUP_RACE_CONTROL_FUNCTION")
    else:
        forced = [
            node for node in ast.walk(control)
            if isinstance(node, ast.Raise)
            and isinstance(node.exc, ast.Call)
            and isinstance(node.exc.func, ast.Name)
            and node.exc.func.id == "ProcessLookupError"
        ]
        if len(forced) != 1:
            errors.append("PROCESS_GROUP_RACE_FORCED_EXCEPTION")
        membership = [
            node for node in ast.walk(control)
            if isinstance(node, ast.Compare)
            and ast.unparse(node) == "members == [valid_pid]"
        ]
        if len(membership) != 1:
            errors.append("PROCESS_GROUP_RACE_EXACT_MEMBERSHIP_ORACLE")
        control_text = ast.unparse(control)
        if control_text.count("producer.Path = controlled_path") != 1 \
                or control_text.count("producer.Path = original_path") != 1:
            errors.append("PROCESS_GROUP_RACE_MONKEYPATCH_RESTORATION")
        if direct_call_count(control, "direct_children_snapshot") != 2:
            errors.append("PROCESS_GROUP_RACE_RESIDUE_ORACLE")
        bounds = [
            node for node in ast.walk(control)
            if isinstance(node, ast.Compare)
            and ast.unparse(node)
            == "elapsed <= PROCESS_GROUP_RACE_TIMEOUT_SECONDS"
        ]
        if len(bounds) != 1:
            errors.append("PROCESS_GROUP_RACE_BOUND_ORACLE")
    if self_test_function is None or direct_call_count(
            self_test_function, "process_group_members_race_control") != 1:
        errors.append("PROCESS_GROUP_RACE_CONTROL_INVOCATION")
    if ctest_function is None \
            or direct_call_count(ctest_function, "self_test") != 1:
        errors.append("PROCESS_GROUP_RACE_CTEST_BINDING")

    producer_function = function_definition(
        producer_tree, "process_group_members")
    if producer_function is None:
        errors.append("PROCESS_GROUP_RACE_PRODUCER_FUNCTION")
    else:
        stat_reads = [
            node for node in ast.walk(producer_function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "read_text"
            and isinstance(node.func.value, ast.BinOp)
            and isinstance(node.func.value.op, ast.Div)
            and isinstance(node.func.value.left, ast.Name)
            and node.func.value.left.id == "item"
            and isinstance(node.func.value.right, ast.Constant)
            and node.func.value.right.value == "stat"
        ]
        containing_tries = [
            node for node in ast.walk(producer_function)
            if isinstance(node, ast.Try)
            and any(read in tuple(ast.walk(statement))
                    for statement in node.body for read in stat_reads)
        ]
        if len(stat_reads) != 1 or len(containing_tries) != 1:
            errors.append("PROCESS_GROUP_RACE_PRODUCER_STAT_BOUNDARY")
        else:
            handlers = containing_tries[0].handlers
            caught: list[str] = []
            if len(handlers) == 1 and isinstance(handlers[0].type, ast.Tuple):
                caught = [
                    item.id for item in handlers[0].type.elts
                    if isinstance(item, ast.Name)
                ]
            expected = [
                "FileNotFoundError", "ProcessLookupError", "PermissionError",
                "ValueError", "IndexError",
            ]
            if caught != expected:
                errors.append("PROCESS_GROUP_RACE_PRODUCER_EXCEPTION_CLAUSE")

    count_node = assignment_value(tree, "REQUIRED_LITERAL_CTEST_REPETITIONS")
    try:
        count_value = ast.literal_eval(count_node) \
            if count_node is not None else None
    except (ValueError, TypeError):
        count_value = None
    if count_value != 24:
        errors.append("LITERAL_CTEST_REPETITION_LITERAL")
    if repetition is None:
        errors.append("LITERAL_CTEST_REPETITION_HELPER")
    elif self_test_function is None or direct_call_count(
            self_test_function, "literal_ctest_repetition_errors") != 2:
        errors.append("LITERAL_CTEST_REPETITION_CONTROLS")
    return errors


def structural_errors(
    source: str, producer_source: str | None = None,
) -> list[str]:
    errors: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return ["VALIDATION_OUTER_SOURCE_SYNTAX:" + str(error)]

    inventory_node = assignment_value(tree, "EXPECTED_EXECUTION_IDS")
    try:
        inventory_value = ast.literal_eval(inventory_node) \
            if inventory_node is not None else None
    except (ValueError, TypeError):
        inventory_value = None
    if inventory_value != EXPECTED_EXECUTION_IDS:
        errors.append("VALIDATION_OUTER_EXPECTED_INVENTORY_LITERAL")
    cardinality_node = assignment_value(
        tree, "EXPECTED_EXECUTION_CARDINALITY")
    try:
        cardinality_value = ast.literal_eval(cardinality_node) \
            if cardinality_node is not None else None
    except (ValueError, TypeError):
        cardinality_value = None
    if cardinality_value != 16:
        errors.append("VALIDATION_OUTER_EXPECTED_CARDINALITY_LITERAL")

    live = function_definition(tree, "live_mode")
    if live is None or direct_call_count(live, "run_live_producer") != 1:
        errors.append("VALIDATION_OUTER_PRODUCER_CALL_CARDINALITY")
    validator = function_definition(tree, "validate_transcript")
    if validator is None:
        errors.append("VALIDATION_OUTER_VALIDATOR_FUNCTION")
    else:
        inventory_comparisons = [
            node for node in ast.walk(validator)
            if isinstance(node, ast.Compare)
            and any(isinstance(operator, ast.NotEq) for operator in node.ops)
            and "set(identities)" in ast.unparse(node)
            and "set(expected_ids)" in ast.unparse(node)
        ]
        if len(inventory_comparisons) != 1:
            errors.append("VALIDATION_OUTER_INVENTORY_COMPARISON_STRUCTURE")
        oracle_blocks = [
            node for node in ast.walk(validator)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "row_oracle_enabled"
            and any(
                isinstance(item, ast.Constant)
                and item.value == "oracle_satisfied"
                for item in ast.walk(node)
            )
        ]
        if len(oracle_blocks) != 1:
            errors.append("VALIDATION_OUTER_ROW_ORACLE_STRUCTURE")

    if producer_source is None:
        producer_source = (
            ROOT / "tools/verify_m018_global_corrections.py"
        ).read_text(encoding="ascii")
    errors.extend(producer_package_oracle_binding_errors(producer_source))
    errors.extend(package_oracle_outer_binding_errors(source))
    try:
        producer_tree = ast.parse(producer_source)
    except SyntaxError as error:
        errors.append("VALIDATION_OUTER_PRODUCER_SOURCE_SYNTAX:" + str(error))
        return errors
    producer_self_test = function_definition(producer_tree, "self_test")
    if producer_self_test is None or direct_call_count(
            producer_self_test, "build_live_transcript") != 1:
        errors.append("VALIDATION_OUTER_PRODUCER_MATRIX_CALL_CARDINALITY")
    errors.extend(process_group_race_structure_errors(
        source, producer_source))

    phases_node = assignment_value(tree, "TIMESTRETCH_FAILURE_PHASES")
    try:
        phases_value = ast.literal_eval(phases_node) \
            if phases_node is not None else None
    except (ValueError, TypeError):
        phases_value = None
    if phases_value != TIMESTRETCH_FAILURE_PHASES:
        errors.append("VALIDATION_OUTER_TIMESTRETCH_PHASE_INVENTORY")
    timestretch = function_definition(tree, "timestretch_live")
    if timestretch is None:
        errors.append("VALIDATION_OUTER_TIMESTRETCH_LIVE_FUNCTION")
    else:
        if direct_call_count(timestretch, "validate_timestretch_rollback") != 1:
            errors.append("VALIDATION_OUTER_TIMESTRETCH_ROLLBACK_CALL")
        phase_loops = [
            node for node in ast.walk(timestretch)
            if isinstance(node, ast.For)
            and isinstance(node.iter, ast.Name)
            and node.iter.id == "TIMESTRETCH_FAILURE_PHASES"
            and ast.unparse(node.target) == "(phase, variable)"
        ]
        position_loops = [
            node for node in ast.walk(timestretch)
            if isinstance(node, ast.For)
            and isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "enumerate"
            and any(isinstance(item, ast.Name)
                    and item.id == "EXPECTED_OUTPUTS"
                    for item in node.iter.args)
        ]
        if len(phase_loops) != 1 or len(position_loops) != 1:
            errors.append("VALIDATION_OUTER_TIMESTRETCH_POSITION_LOOPS")
    rollback_validator = function_definition(
        tree, "validate_timestretch_rollback")
    if rollback_validator is None:
        errors.append("VALIDATION_OUTER_TIMESTRETCH_ROLLBACK_ORACLE")
    else:
        validator_text = ast.unparse(rollback_validator)
        if "after != before" not in validator_text:
            errors.append("VALIDATION_OUTER_TIMESTRETCH_SENTINEL_ORACLE")
        if "timestretch_transaction_residue(directory)" \
                not in validator_text:
            errors.append("VALIDATION_OUTER_TIMESTRETCH_CLEANUP_ORACLE")
    errors.extend(timestretch_durable_structure_errors(source))
    errors.extend(timestretch_coherence_structure_errors(
        source,
        (ROOT / "tools/characterize_timestretch.cpp")
        .read_text(encoding="ascii")))
    return errors


def timestretch_durable_structure_errors(source: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return ["TIMESTRETCH_DURABLE_SOURCE_SYNTAX:" + str(error)]
    errors: list[str] = []
    for name, expected, cardinality, terminal in (
        ("TIMESTRETCH_TRANSACTION_STATES", TIMESTRETCH_TRANSACTION_STATES,
         13, "TIMESTRETCH_DURABLE_STATE_INVENTORY"),
        ("TIMESTRETCH_TRANSACTION_TRANSITIONS",
         TIMESTRETCH_TRANSACTION_TRANSITIONS, 28,
         "TIMESTRETCH_DURABLE_TRANSITION_INVENTORY"),
    ):
        node = assignment_value(tree, name)
        try:
            value = ast.literal_eval(node) if node is not None else None
        except (ValueError, TypeError):
            value = None
        if value != expected or len(value or ()) != cardinality:
            errors.append(terminal)
    fault_node = assignment_value(tree, "TIMESTRETCH_DURABLE_FAULT_POINTS")
    try:
        fault_value = ast.literal_eval(fault_node) \
            if isinstance(fault_node, ast.Tuple) else None
    except (ValueError, TypeError):
        fault_value = None
    if fault_value != TIMESTRETCH_DURABLE_FAULT_POINTS \
            or len(fault_value or ()) != 193:
        errors.append("TIMESTRETCH_DURABLE_LITERAL_FAULT_INVENTORY")

    runner = function_definition(tree, "run_timestretch_durable_matrix")
    live = function_definition(tree, "timestretch_live")
    if runner is None or live is None \
            or direct_call_count(live, "run_timestretch_durable_matrix") != 1:
        errors.append("TIMESTRETCH_DURABLE_RECOVERY_INVOCATION")
    if runner is not None:
        loops = {
            ast.unparse(node.iter): node
            for node in ast.walk(runner) if isinstance(node, ast.For)
        }
        if "TIMESTRETCH_CRASH_POINTS" not in loops:
            errors.append("TIMESTRETCH_DURABLE_CRASH_LOOP")
        if "TIMESTRETCH_JOURNAL_MUTATIONS" not in loops:
            errors.append("TIMESTRETCH_DURABLE_JOURNAL_MUTATION_LOOP")
        if direct_call_count(runner, "is_recovery_required") != 2:
            errors.append("TIMESTRETCH_DURABLE_RECOVERY_TERMINAL_ORACLE")
    clean = function_definition(tree, "durable_process_clean")
    success_suppression = [] if clean is None else [
        node for node in ast.walk(clean)
        if isinstance(node, ast.Compare)
        and any(isinstance(operator, ast.NotIn) for operator in node.ops)
        and "characterisation written" in ast.unparse(node)
        and "stdout" in ast.unparse(node)
    ]
    if len(success_suppression) != 1:
        errors.append("TIMESTRETCH_DURABLE_SUCCESS_SUPPRESSION_ORACLE")
    if live is None:
        errors.append("TIMESTRETCH_DURABLE_OBSERVED_IDENTITY_COMPARISON")
        errors.append("TIMESTRETCH_DURABLE_LEGACY_LOOP")
    else:
        comparisons = [
            node for node in ast.walk(live)
            if isinstance(node, ast.Compare)
            and any(isinstance(operator, ast.NotEq) for operator in node.ops)
            and "EXPECTED_TIMESTRETCH_EXECUTION_IDS" in ast.unparse(node)
        ]
        if len(comparisons) != 1:
            errors.append("TIMESTRETCH_DURABLE_OBSERVED_IDENTITY_COMPARISON")
        legacy_loops = [
            node for node in ast.walk(live)
            if isinstance(node, ast.For)
            and isinstance(node.iter, ast.Name)
            and node.iter.id == "TIMESTRETCH_FAILURE_PHASES"
            and ast.unparse(node.target) == "(phase, variable)"
        ]
        if len(legacy_loops) != 1:
            errors.append("TIMESTRETCH_DURABLE_LEGACY_LOOP")
    return errors


def timestretch_coherence_structure_errors(
    source: str, characterizer_source: str,
) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return ["TIMESTRETCH_COHERENCE_SOURCE_SYNTAX:" + str(error)]
    errors: list[str] = []
    transition_node = assignment_value(
        tree, "TIMESTRETCH_COHERENCE_TRANSITIONS")
    pair_node = assignment_value(tree, "TIMESTRETCH_COHERENCE_PAIR_CASES")
    metadata_node = assignment_value(
        tree, "TIMESTRETCH_COHERENCE_METADATA_CASES")
    outer_node = assignment_value(
        tree, "TIMESTRETCH_COHERENCE_OUTER_MUTANTS")
    try:
        transition_value = ast.literal_eval(transition_node) \
            if transition_node is not None else None
        pair_value = ast.literal_eval(pair_node) \
            if pair_node is not None else None
        metadata_value = ast.literal_eval(metadata_node) \
            if metadata_node is not None else None
        outer_value = ast.literal_eval(outer_node) \
            if outer_node is not None else None
    except (ValueError, TypeError):
        transition_value = pair_value = metadata_value = outer_value = None
    if transition_value != TIMESTRETCH_COHERENCE_TRANSITIONS \
            or len(transition_value or ()) != 32:
        errors.append("TIMESTRETCH_COHERENCE_TRANSITION_INVENTORY")
    if pair_value != TIMESTRETCH_COHERENCE_PAIR_CASES \
            or len(pair_value or ()) != 36:
        errors.append("TIMESTRETCH_COHERENCE_PAIR_INVENTORY")
    if metadata_value != TIMESTRETCH_COHERENCE_METADATA_CASES \
            or len(metadata_value or ()) != 18:
        errors.append("TIMESTRETCH_COHERENCE_METADATA_INVENTORY")
    if outer_value != TIMESTRETCH_COHERENCE_OUTER_MUTANTS \
            or len(outer_value or ()) != 40:
        errors.append("TIMESTRETCH_COHERENCE_OUTER_INVENTORY")
    runner = function_definition(tree, "run_timestretch_coherence_matrix")
    live = function_definition(tree, "timestretch_live")
    if runner is None or live is None or direct_call_count(
            live, "run_timestretch_coherence_matrix") != 1:
        errors.append("TIMESTRETCH_COHERENCE_LIVE_INVOCATION")
    if runner is None:
        errors.extend((
            "TIMESTRETCH_COHERENCE_PAIR_LOOP",
            "TIMESTRETCH_COHERENCE_TRANSITION_LOOP",
            "TIMESTRETCH_COHERENCE_METADATA_LOOP",
        ))
    else:
        loop_names = {
            ast.unparse(node.iter) for node in ast.walk(runner)
            if isinstance(node, ast.For)
        }
        for name, terminal in (
            ("TIMESTRETCH_COHERENCE_PAIR_CASES",
             "TIMESTRETCH_COHERENCE_PAIR_LOOP"),
            ("TIMESTRETCH_COHERENCE_TRANSITIONS",
             "TIMESTRETCH_COHERENCE_TRANSITION_LOOP"),
            ("TIMESTRETCH_COHERENCE_METADATA_CASES",
             "TIMESTRETCH_COHERENCE_METADATA_LOOP"),
        ):
            if name not in loop_names:
                errors.append(terminal)
    checker = function_definition(
        tree, "timestretch_coherence_structure_errors")
    transition_comparisons = [] if checker is None else [
        node for node in ast.walk(checker)
        if isinstance(node, ast.Compare)
        and any(isinstance(operator, ast.NotEq) for operator in node.ops)
        and isinstance(node.left, ast.Name)
        and node.left.id == "transition_value"
        and any(isinstance(item, ast.Name)
                and item.id == "TIMESTRETCH_COHERENCE_TRANSITIONS"
                for item in node.comparators)
    ]
    if len(transition_comparisons) != 1:
        errors.append("TIMESTRETCH_COHERENCE_TRANSITION_INVENTORY_COMPARISON")
    execution_comparisons = [] if live is None else [
        node for node in ast.walk(live)
        if isinstance(node, ast.Compare)
        and any(isinstance(operator, ast.NotEq) for operator in node.ops)
        and isinstance(node.left, ast.Name) and node.left.id == "observed_rows"
        and "EXPECTED_TIMESTRETCH_EXECUTION_ROWS" in ast.unparse(node)
    ]
    if len(execution_comparisons) != 1:
        errors.append("TIMESTRETCH_COHERENCE_EXECUTION_INVENTORY_COMPARISON")
    if characterizer_source.count("bool transactionClassifyAuthority(") != 1:
        errors.append("TIMESTRETCH_COHERENCE_PAIR_VALIDATOR")
    if "high.journal.previousPayloadSha256 != low.payloadSha256" \
            not in characterizer_source:
        errors.append("TIMESTRETCH_COHERENCE_PREDECESSOR_BINDING")
    if "transactionJournalBodyEqual(high, supplied)" \
            not in characterizer_source:
        errors.append("TIMESTRETCH_COHERENCE_METADATA_COMPARISON")
    return errors


def threading_control_structure_errors(source: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return ["THREADING_CONTROL_SOURCE_SYNTAX:" + str(error)]
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "build_census_controls"
    ]
    loops = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and isinstance(node.iter, ast.Name)
        and node.iter.id == "census_controls"
    ]
    errors: list[str] = []
    if len(calls) != 1:
        errors.append("THREADING_INTERNAL_CONTROL_INVOCATION")
    if len(loops) != 1:
        errors.append("THREADING_INTERNAL_CONTROL_EXECUTION")
    baseline_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "validate_pin_census"
        and any(isinstance(argument, ast.Name) and argument.id == "doc"
                for argument in node.args)
    ]
    if len(calls) == 1 and (len(baseline_calls) != 1
                            or baseline_calls[0].lineno >= calls[0].lineno):
        errors.append("THREADING_BASELINE_BEFORE_INTERNAL_CONTROLS")
    return errors


GCC_DRIVER_TOKEN = (
    r"(?<![A-Za-z0-9_+.-])"
    r"(?:g\+\+(?:-\d+)?|gcc(?:-\d+)?)"
    r"(?![A-Za-z0-9_+.-])"
)


def compiler_version_identity(first_line: str) -> tuple[str | None, int | None]:
    clang = re.search(r"\bclang version\s+(\d+)(?:\.|\b)", first_line, re.I)
    if clang:
        return "clang", int(clang.group(1))
    if "clang" in first_line.lower():
        return "clang", None
    gcc = re.search(
        GCC_DRIVER_TOKEN + r".*?"
        r"(?:\)\s*|\bversion\s+)?(\d+)\.(\d+)", first_line, re.I)
    if gcc:
        return "gcc", int(gcc.group(1))
    if re.search(GCC_DRIVER_TOKEN, first_line, re.I):
        return "gcc", None
    return None, None


def process_record_errors(
    record: object, prefix: str, expected_exit: int,
    expected_command: list[str] | None = None,
) -> list[str]:
    if not isinstance(record, dict):
        return [prefix + ":TYPE"]
    errors: list[str] = []
    if expected_command is not None and record.get("command") != expected_command:
        errors.append(prefix + ":COMMAND")
    if record.get("exit_code") != expected_exit:
        errors.append(prefix + ":EXIT")
    if record.get("timed_out") is not False:
        errors.append(prefix + ":TIMEOUT")
    if record.get("errors") != []:
        errors.append(prefix + ":ERRORS")
    if record.get("cleanup_residue"):
        errors.append(prefix + ":RESIDUE")
    if record.get("residue_detected"):
        errors.append(prefix + ":RESIDUE_DETECTED")
    for stream in ("stdout", "stderr"):
        value = record.get(stream)
        if not isinstance(value, str):
            errors.append(prefix + ":" + stream.upper() + "_TYPE")
            continue
        encoded = value.encode("utf-8")
        if len(encoded) > MAX_CAPTURE_BYTES:
            errors.append(prefix + ":" + stream.upper() + "_LIMIT")
        if record.get(stream + "_size") != len(encoded):
            errors.append(prefix + ":" + stream.upper() + "_SIZE")
        if record.get(stream + "_sha256") != hashlib.sha256(encoded).hexdigest():
            errors.append(prefix + ":" + stream.upper() + "_HASH")
    return errors


def validate_transcript(
    transcript: dict[str, object], producer_exit: int = 0,
    expected_ids: tuple[str, ...] = EXPECTED_EXECUTION_IDS,
    comparison_enabled: bool = True, row_oracle_enabled: bool = True,
) -> list[str]:
    errors: list[str] = []
    if producer_exit != 0:
        errors.append("VALIDATION_OUTER_PRODUCER_EXIT:{}".format(producer_exit))
    if transcript.get("schema") != "dspark.global-validation-producer.v1":
        errors.append("VALIDATION_OUTER_TRANSCRIPT_SCHEMA")
    if transcript.get("status") != "PASS" or transcript.get("errors") != []:
        errors.append("VALIDATION_OUTER_TRANSCRIPT_STATUS")
    if transcript.get("invocation_count") != 1:
        errors.append("VALIDATION_OUTER_MATRIX_INVOCATION_COUNT:{}".format(
            transcript.get("invocation_count")))
    if len(expected_ids) != EXPECTED_EXECUTION_CARDINALITY:
        errors.append("VALIDATION_OUTER_EXPECTED_INVENTORY_CARDINALITY:{}".format(
            len(expected_ids)))
    if len(set(expected_ids)) != len(expected_ids):
        errors.append("VALIDATION_OUTER_EXPECTED_INVENTORY_DUPLICATE")
    if not comparison_enabled:
        errors.append("VALIDATION_OUTER_INVENTORY_COMPARISON_DISABLED")
    if not row_oracle_enabled:
        errors.append("VALIDATION_OUTER_ROW_ORACLE_DISABLED")

    expected_discovery = {"gcc13": ("gcc", 13), "clang18": ("clang", 18)}
    discovery = transcript.get("compiler_discovery")
    observed: dict[object, dict[str, object]] = {}
    if not isinstance(discovery, list) or len(discovery) != 2 \
            or not all(isinstance(item, dict) for item in discovery):
        errors.append("VALIDATION_OUTER_DISCOVERY_CARDINALITY")
    else:
        observed = {item.get("label"): item for item in discovery}
        if set(observed) != set(expected_discovery):
            errors.append("VALIDATION_OUTER_DISCOVERY_IDENTITIES")
        for label, (family, major) in expected_discovery.items():
            item = observed.get(label, {})
            if (item.get("status"), item.get("family"), item.get("major")) \
                    != ("PASS", family, major):
                errors.append("VALIDATION_OUTER_DISCOVERY_ORACLE:" + label)
            path = item.get("path")
            resolved = item.get("resolved_path")
            if not isinstance(path, str) or not path \
                    or not isinstance(resolved, str) or not resolved:
                errors.append("VALIDATION_OUTER_DISCOVERY_PATH:" + label)
                continue
            version = item.get("version_process")
            dump = item.get("dump_process")
            errors.extend(process_record_errors(
                version, "VALIDATION_OUTER_VERSION_PROCESS:" + label, 0,
                [path, "--version"]))
            errors.extend(process_record_errors(
                dump, "VALIDATION_OUTER_DUMP_PROCESS:" + label, 0,
                [path, "-dumpfullversion", "-dumpversion"]))
            first_line = item.get("version_first_line")
            if not isinstance(first_line, str) \
                    or compiler_version_identity(first_line) != (family, major):
                errors.append("VALIDATION_OUTER_VERSION_IDENTITY:" + label)
            dumped = item.get("dump_version")
            match = re.fullmatch(r"\s*(\d+)(?:\.\d+){0,3}\s*", dumped) \
                if isinstance(dumped, str) else None
            if match is None or int(match.group(1)) != major:
                errors.append("VALIDATION_OUTER_DUMP_IDENTITY:" + label)

    sources = transcript.get("source_variants")
    if not isinstance(sources, list) or len(sources) != 4:
        errors.append("VALIDATION_OUTER_SOURCE_CARDINALITY")
        source_by_id: dict[str, dict[str, object]] = {}
    else:
        source_by_id = {
            str(item.get("id")): item for item in sources
            if isinstance(item, dict)
        }
        if len(source_by_id) != 4 or set(source_by_id) != set(EXPECTED_SOURCE_HASHES):
            errors.append("VALIDATION_OUTER_SOURCE_IDENTITIES")
        for variant, expected_hash in EXPECTED_SOURCE_HASHES.items():
            item = source_by_id.get(variant, {})
            if item.get("source_sha256") != expected_hash:
                errors.append("VALIDATION_OUTER_SOURCE_HASH:" + variant)
            if item.get("insertion_sha256") != EXPECTED_INSERTION_HASHES[variant]:
                errors.append("VALIDATION_OUTER_INSERTION_HASH:" + variant)
            if item.get("patch_sha256") != EXPECTED_PATCH_HASHES[variant]:
                errors.append("VALIDATION_OUTER_PATCH_HASH:" + variant)
            if item.get("anchor_count_before") != 1 \
                    or item.get("anchor_count_after") != 1:
                errors.append("VALIDATION_OUTER_SOURCE_ANCHOR:" + variant)
            expected_delta = 0 if variant == "baseline" else 1
            if item.get("insertion_delta") != expected_delta:
                errors.append("VALIDATION_OUTER_SOURCE_INSERTION:" + variant)

    rows = transcript.get("executions")
    if not isinstance(rows, list):
        rows = []
        errors.append("VALIDATION_OUTER_EXECUTIONS_TYPE")
    identities = [row.get("id") for row in rows if isinstance(row, dict)]
    if len(identities) != len(rows):
        errors.append("VALIDATION_OUTER_ROW_TYPE")
    if len(identities) != len(set(identities)):
        errors.append("VALIDATION_OUTER_OBSERVED_IDENTITY_DUPLICATE")
    if not comparison_enabled:
        pass
    elif set(identities) != set(expected_ids):
        errors.append("VALIDATION_OUTER_MATRIX_IDENTITY_SET")
    contracts = {
        identity: (compiler, profile, variant, oracle)
        for identity, compiler, profile, variant, oracle in EXPECTED_ROWS
    }
    if row_oracle_enabled:
        for row in rows:
            if not isinstance(row, dict):
                continue
            identity = row.get("id")
            expected = contracts.get(identity)
            if expected is None:
                errors.append("VALIDATION_OUTER_UNEXPECTED_ROW:{}".format(identity))
                continue
            fields = tuple(row.get(name) for name in
                           ("compiler", "profile", "variant", "oracle"))
            if fields != expected:
                errors.append("VALIDATION_OUTER_ROW_FIELDS:" + str(identity))
            compiler, profile, variant, oracle = expected
            if row.get("build_ok") is not True:
                errors.append("VALIDATION_OUTER_BUILD_NOT_OK:" + str(identity))
            if row.get("run_timed_out") is not False:
                errors.append("VALIDATION_OUTER_RUN_TIMEOUT:" + str(identity))
            if row.get("cleanup_residue") != []:
                errors.append("VALIDATION_OUTER_CLEANUP_RESIDUE:" + str(identity))
            if row.get("oracle_satisfied") is not True:
                errors.append("VALIDATION_OUTER_ROW_ORACLE:" + str(identity))
            if row.get("source_sha256") != EXPECTED_SOURCE_HASHES[variant]:
                errors.append("VALIDATION_OUTER_ROW_SOURCE:" + str(identity))

            compile_record = row.get("compile")
            errors.extend(process_record_errors(
                compile_record, "VALIDATION_OUTER_COMPILE:" + str(identity), 0))
            command = compile_record.get("command", []) \
                if isinstance(compile_record, dict) else []
            if not isinstance(command, list):
                command = []
            discovered_path = observed.get(compiler, {}).get("path")
            if not command or command[0] != discovered_path:
                errors.append("VALIDATION_OUTER_COMPILE_DRIVER:" + str(identity))
            for flag in COMMON_COMPILE_FLAGS:
                if command.count(flag) != 1:
                    errors.append("VALIDATION_OUTER_COMPILE_FLAG:{}:{}".format(
                        identity, flag))
            for flag in SANITIZER_FLAGS:
                count = command.count(flag)
                if (profile == "sanitizer" and count != 1) \
                        or (profile == "normal" and count != 0):
                    errors.append("VALIDATION_OUTER_PROFILE_FLAG:{}:{}".format(
                        identity, flag))
            gcc_flag_count = command.count("-Wno-mismatched-new-delete")
            if (compiler == "gcc13" and gcc_flag_count != 1) \
                    or (compiler == "clang18" and gcc_flag_count != 0):
                errors.append("VALIDATION_OUTER_COMPILER_FLAG:" + str(identity))
            if not any(str(item).endswith("/tests/TestReverbPublication.cpp")
                       for item in command) \
                    or command.count("-I") != 1 or command.count("-o") != 1:
                errors.append("VALIDATION_OUTER_COMPILE_SUBJECT:" + str(identity))
            if isinstance(compile_record, dict) \
                    and (compile_record.get("stdout") != ""
                         or compile_record.get("stderr") != ""):
                errors.append("VALIDATION_OUTER_COMPILE_OUTPUT:" + str(identity))

            expected_environment = NORMAL_RUNTIME_ENVIRONMENT \
                if profile == "normal" else SANITIZER_RUNTIME_ENVIRONMENT
            environment_contract = row.get("environment_contract")
            if environment_contract != {
                    "remove_first": list(REMOVED_RUNTIME_ENVIRONMENT),
                    "set": expected_environment}:
                errors.append("VALIDATION_OUTER_RUNTIME_ENVIRONMENT:" + str(identity))

            expected_exit, expected_stdout, expected_stderr = \
                EXPECTED_OUTPUT_CONTRACTS[oracle]
            run_record = row.get("run")
            errors.extend(process_record_errors(
                run_record, "VALIDATION_OUTER_RUN:" + str(identity),
                expected_exit))
            if isinstance(run_record, dict):
                stdout = run_record.get("stdout")
                stderr = run_record.get("stderr")
                if stdout != expected_stdout or stderr != expected_stderr:
                    errors.append("VALIDATION_OUTER_RUN_OUTPUT:" + str(identity))
                combined = str(stdout) + str(stderr)
                if any(marker in combined for marker in SANITIZER_DIAGNOSTICS):
                    errors.append("VALIDATION_OUTER_SANITIZER_DIAGNOSTIC:" + str(identity))
    dict_rows = [row for row in rows if isinstance(row, dict)]
    if len([row for row in dict_rows if row.get("oracle") == "GREEN"]) != 4:
        errors.append("VALIDATION_OUTER_BASELINE_CARDINALITY")
    if len([row for row in dict_rows
            if row.get("oracle") == "EXPECTED_RED"]) != 12:
        errors.append("VALIDATION_OUTER_EXPECTED_RED_CARDINALITY")
    return errors


def synthetic_process(
    command: list[str], exit_code: int, stdout: str = "", stderr: str = "",
) -> dict[str, object]:
    stdout_bytes = stdout.encode("utf-8")
    stderr_bytes = stderr.encode("utf-8")
    return {
        "command": command,
        "exit_code": exit_code,
        "timed_out": False,
        "stdout": stdout,
        "stderr": stderr,
        "stdout_sha256": hashlib.sha256(stdout_bytes).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr_bytes).hexdigest(),
        "stdout_size": len(stdout_bytes),
        "stderr_size": len(stderr_bytes),
        "cleanup_residue": [],
        "residue_detected": [],
        "errors": [],
    }


def synthetic_transcript() -> dict[str, object]:
    sources = [
        {
            "id": variant,
            "source_sha256": source_hash,
            "insertion_sha256": EXPECTED_INSERTION_HASHES[variant],
            "patch_sha256": EXPECTED_PATCH_HASHES[variant],
            "anchor_count_before": 1,
            "anchor_count_after": 1,
            "insertion_delta": 0 if variant == "baseline" else 1,
        }
        for variant, source_hash in EXPECTED_SOURCE_HASHES.items()
    ]
    rows = []
    for identity, compiler, profile, variant, oracle in EXPECTED_ROWS:
        driver = "/opt/g++-13" if compiler == "gcc13" \
            else "/opt/clang++-18"
        flags = list(COMMON_COMPILE_FLAGS)
        if profile == "sanitizer":
            flags.extend(SANITIZER_FLAGS)
        if compiler == "gcc13":
            flags.append("-Wno-mismatched-new-delete")
        source_root = "/tmp/" + variant
        command = [
            driver, *flags, "-I", source_root,
            source_root + "/tests/TestReverbPublication.cpp",
            "-o", source_root + "/subject",
        ]
        expected_exit, expected_stdout, expected_stderr = \
            EXPECTED_OUTPUT_CONTRACTS[oracle]
        environment = NORMAL_RUNTIME_ENVIRONMENT \
            if profile == "normal" else SANITIZER_RUNTIME_ENVIRONMENT
        rows.append({
            "id": identity,
            "compiler": compiler,
            "profile": profile,
            "variant": variant,
            "oracle": oracle,
            "source_sha256": EXPECTED_SOURCE_HASHES[variant],
            "build_ok": True,
            "run_timed_out": False,
            "cleanup_residue": [],
            "oracle_satisfied": True,
            "environment_contract": {
                "remove_first": list(REMOVED_RUNTIME_ENVIRONMENT),
                "set": environment,
            },
            "compile": synthetic_process(command, 0),
            "run": synthetic_process(
                [source_root + "/subject"], expected_exit,
                expected_stdout, expected_stderr),
        })
    return {
        "schema": "dspark.global-validation-producer.v1",
        "status": "PASS",
        "errors": [],
        "invocation_count": 1,
        "compiler_discovery": [
            {
                "label": "gcc13", "status": "PASS", "family": "gcc",
                "major": 13,
                "path": "/opt/g++-13",
                "resolved_path": "/opt/g++-13",
                "version_first_line": "g++-13 (control) 13.3.0",
                "dump_version": "13.3.0",
                "version_process": synthetic_process(
                    ["/opt/g++-13", "--version"], 0,
                    "g++-13 (control) 13.3.0\n"),
                "dump_process": synthetic_process(
                    ["/opt/g++-13", "-dumpfullversion", "-dumpversion"],
                    0, "13.3.0\n"),
            },
            {
                "label": "clang18", "status": "PASS", "family": "clang",
                "major": 18,
                "path": "/opt/clang++-18",
                "resolved_path": "/opt/clang++-18",
                "version_first_line": "clang version 18.1.3",
                "dump_version": "18.1.3",
                "version_process": synthetic_process(
                    ["/opt/clang++-18", "--version"], 0,
                    "clang version 18.1.3\n"),
                "dump_process": synthetic_process(
                    ["/opt/clang++-18", "-dumpfullversion", "-dumpversion"],
                    0, "18.1.3\n"),
            },
        ],
        "source_variants": sources,
        "executions": rows,
    }


def literal_ctest_repetition_errors(
    records: list[dict[str, object]],
) -> list[str]:
    errors: list[str] = []
    if len(records) < REQUIRED_LITERAL_CTEST_REPETITIONS:
        errors.append("LITERAL_CTEST_REPEAT_CARDINALITY:{}".format(
            len(records)))
    for index, record in enumerate(records, 1):
        prefix = "LITERAL_CTEST_REPEAT_{:02d}".format(index)
        if record.get("command") != list(LITERAL_CTEST_COMMAND):
            errors.append(prefix + ":COMMAND")
        if record.get("exit_code") != 0:
            errors.append(prefix + ":EXIT")
        if record.get("timed_out") is not False:
            errors.append(prefix + ":TIMEOUT")
        if record.get("stderr") != "":
            errors.append(prefix + ":STDERR")
        if record.get("errors") != []:
            errors.append(prefix + ":ERRORS")
        duration = record.get("duration_seconds")
        if not isinstance(duration, (int, float)) or duration < 0 \
                or duration > LITERAL_CTEST_TIMEOUT_SECONDS:
            errors.append(prefix + ":BOUND")
        stdout = record.get("stdout")
        if not isinstance(stdout, str) \
                or not stdout.endswith(
                    "PASS global validation contract ctest\n"):
            errors.append(prefix + ":STDOUT")
    return errors


def direct_children_snapshot() -> tuple[int, ...] | None:
    children = (
        Path("/proc") / str(os.getpid()) / "task" / str(os.getpid())
        / "children"
    )
    try:
        fields = children.read_text(encoding="ascii").split()
        return tuple(sorted(int(field) for field in fields))
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
        return None


def process_group_members_race_control() -> list[tuple[str, bool]]:
    if os.name != "posix" or not Path("/proc").is_dir():
        return [("process-group-race-platform-deferred", True)]

    target_group = 246_810
    vanished_pid = 41001
    valid_pid = 41002
    zombie_pid = 41003
    foreign_pid = 41004

    class PseudoStat:
        def __init__(self, text: str, vanished: bool = False) -> None:
            self.text = text
            self.vanished = vanished
            self.read_count = 0

        def read_text(self, encoding: str) -> str:
            self.read_count += 1
            if encoding != "ascii":
                raise ValueError("controlled stat encoding")
            if self.vanished:
                raise ProcessLookupError("controlled vanished /proc entry")
            return self.text

    class PseudoEntry:
        def __init__(self, pid: int, stat_file: PseudoStat) -> None:
            self.name = str(pid)
            self.stat_file = stat_file

        def __truediv__(self, name: str) -> PseudoStat:
            if name != "stat":
                raise ValueError("controlled stat path")
            return self.stat_file

    vanished = PseudoStat("", vanished=True)
    valid = PseudoStat(
        "{} (dspark valid) S 1 {} 1 0\n".format(
            valid_pid, target_group))
    zombie = PseudoStat(
        "{} (dspark zombie) Z 1 {} 1 0\n".format(
            zombie_pid, target_group))
    foreign = PseudoStat(
        "{} (dspark foreign) S 1 {} 1 0\n".format(
            foreign_pid, target_group + 1))
    entries = (
        PseudoEntry(vanished_pid, vanished),
        PseudoEntry(valid_pid, valid),
        PseudoEntry(zombie_pid, zombie),
        PseudoEntry(foreign_pid, foreign),
    )

    class PseudoProc:
        @staticmethod
        def is_dir() -> bool:
            return True

        @staticmethod
        def iterdir() -> tuple[PseudoEntry, ...]:
            return entries

    original_path = producer.Path
    pseudo_proc = PseudoProc()

    def controlled_path(value: object) -> object:
        return pseudo_proc if os.fspath(value) == "/proc" \
            else original_path(value)

    children_before = direct_children_snapshot()
    started = time.monotonic()
    producer.Path = controlled_path
    try:
        members = producer.process_group_members(target_group)
    finally:
        producer.Path = original_path
    elapsed = time.monotonic() - started
    children_after = direct_children_snapshot()
    exact_members = members == [valid_pid]
    exact_reads = all(item.stat_file.read_count == 1 for item in entries)
    return [
        ("process-group-race-forced-vanish", vanished.read_count == 1),
        ("process-group-race-exact-members", exact_members),
        ("process-group-race-all-entries-read", exact_reads),
        ("process-group-race-monkeypatch-restored",
         producer.Path is original_path),
        ("process-group-race-no-process-residue",
         children_before == () and children_after == ()),
        ("process-group-race-bounded",
         elapsed <= PROCESS_GROUP_RACE_TIMEOUT_SECONDS),
    ]


def fake_compiler(path: Path, family: str = "gcc", major: int = 13,
                  mode: str = "ok", driver: str | None = None,
                  dump_major: int | None = None) -> None:
    driver = driver or "gcc"
    first = "{} (validation control) {}.2.0".format(driver, major) \
        if family == "gcc" else "clang version {}.0.1".format(major)
    body = """#!/usr/bin/env python3
import signal
import subprocess
import sys
import time
mode = {mode!r}
if mode == "hang-leader":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    time.sleep(60)
if mode == "hang-descendant":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    subprocess.Popen([sys.executable, "-c", "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"])
    time.sleep(60)
if mode == "residue":
    subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
    )
    raise SystemExit(0)
if mode == "nonzero":
    print({first!r})
    raise SystemExit(9)
if mode == "empty":
    raise SystemExit(0)
if mode == "malformed":
    print("compiler version unknown")
    raise SystemExit(0)
if "--version" in sys.argv:
    print({first!r})
elif "-dumpfullversion" in sys.argv:
    if mode == "dump-nonzero":
        raise SystemExit(9)
    if mode == "dump-empty":
        raise SystemExit(0)
    if mode == "dump-malformed":
        print("unknown")
        raise SystemExit(0)
    if mode == "dump-hang":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        time.sleep(60)
    print({dump!r})
else:
    time.sleep(60)
""".format(
        mode=mode,
        first=first,
        dump="{}.2.0".format(major if dump_major is None else dump_major),
    )
    path.write_text(body, encoding="ascii")
    path.chmod(0o755)


def compiler_process_controls() -> list[tuple[str, bool]]:
    controls: list[tuple[str, bool]] = []
    if os.name != "posix" or not Path("/proc").is_dir():
        return [("compiler-process-controls-platform-deferred", True)]
    with tempfile.TemporaryDirectory(
            prefix="dspark-compiler-controls-") as directory:
        scratch = Path(directory)
        old_timeout = producer.VERSION_TIMEOUT_SECONDS
        old_grace = producer.PROCESS_GRACE_SECONDS
        old_cleanup = producer.PROCESS_CLEANUP_SECONDS
        producer.VERSION_TIMEOUT_SECONDS = 0.25
        producer.PROCESS_GRACE_SECONDS = 0.1
        producer.PROCESS_CLEANUP_SECONDS = 0.5
        try:
            for name, driver in (("generic-gxx", "g++"),
                                 ("versioned-gxx", "g++-13")):
                path = scratch / ("compiler-" + name)
                fake_compiler(path, driver=driver)
                record = producer.discover_compiler(
                    name, "gcc", 13, (driver,), scratch,
                    {driver: str(path)})
                controls.append((
                    "compiler-" + name + "-positive",
                    record.get("status") == "PASS"
                    and record.get("candidate") == driver
                    and record.get("family") == "gcc"
                    and record.get("major") == 13))
            missing = producer.discover_compiler(
                "gcc13", "gcc", 13, ("candidate",), scratch, {})
            controls.append((
                "compiler-missing",
                "MISSING_EXECUTABLE" in str(missing.get("terminal"))))
            cases = (
                ("compiler-swapped", "gcc", 13, "clang", 18, "FAMILY_MISMATCH"),
                ("compiler-gcc12", "gcc", 13, "gcc", 12, "MAJOR_MISMATCH:12"),
                ("compiler-gcc14", "gcc", 13, "gcc", 14, "MAJOR_MISMATCH:14"),
                ("compiler-clang17", "clang", 18, "clang", 17, "MAJOR_MISMATCH:17"),
                ("compiler-clang19", "clang", 18, "clang", 19, "MAJOR_MISMATCH:19"),
            )
            for name, required_family, required_major, family, major, reason in cases:
                path = scratch / name
                fake_compiler(path, family, major)
                record = producer.discover_compiler(
                    name, required_family, required_major, ("candidate",),
                    scratch, {"candidate": str(path)})
                controls.append((name, reason in str(record.get("terminal"))))
            for mode, reason in (
                    ("empty", "VERSION_OUTPUT_EMPTY"),
                    ("malformed", "VERSION_OUTPUT_MALFORMED"),
                    ("nonzero", "VERSION_PROBE_NONZERO"),
                    ("hang-leader", "VERSION_PROBE_TIMEOUT"),
                    ("hang-descendant", "VERSION_PROBE_TIMEOUT")):
                path = scratch / ("compiler-" + mode)
                fake_compiler(path, mode=mode)
                record = producer.discover_compiler(
                    mode, "gcc", 13, ("candidate",), scratch,
                    {"candidate": str(path)})
                controls.append(("compiler-" + mode,
                                 reason in str(record.get("terminal"))))
            for mode, dump_major, reason in (
                    ("dump-empty", None, "DUMP_VERSION_MALFORMED"),
                    ("dump-malformed", None, "DUMP_VERSION_MALFORMED"),
                    ("dump-nonzero", None, "DUMP_VERSION_NONZERO:9"),
                    ("dump-hang", None, "DUMP_VERSION_TIMEOUT"),
                    ("ok", 14, "DUMP_VERSION_MAJOR_MISMATCH:14")):
                name = "compiler-wrong-dump-{}-{}".format(
                    mode, "default" if dump_major is None else dump_major)
                path = scratch / name
                fake_compiler(path, mode=mode, dump_major=dump_major)
                record = producer.discover_compiler(
                    name, "gcc", 13, ("candidate",), scratch,
                    {"candidate": str(path)})
                controls.append((name,
                                 reason in str(record.get("terminal"))))
            hanging = scratch / "phase-timeout"
            fake_compiler(hanging, mode="hang-leader")
            for phase in ("compile", "run"):
                result = producer.run_owned(
                    [str(hanging)], scratch, 0.1, phase)
                controls.append((
                    phase + "-timeout-cleanup",
                    result["timed_out"] and not result["cleanup_residue"]))
            residue = scratch / "phase-residue"
            fake_compiler(residue, mode="residue")
            result = producer.run_owned([str(residue)], scratch, 1, "run")
            controls.append((
                "process-residue-rejected",
                "PROCESS_TREE_CLEANUP_FAILED" in result["errors"]
                and not result["cleanup_residue"]))
        finally:
            producer.VERSION_TIMEOUT_SECONDS = old_timeout
            producer.PROCESS_GRACE_SECONDS = old_grace
            producer.PROCESS_CLEANUP_SECONDS = old_cleanup
    return controls


def compiler_banner_controls() -> list[tuple[str, bool]]:
    cases = (
        ("generic-gxx", "g++ (Ubuntu 13.4.0) 13.4.0", ("gcc", 13)),
        ("versioned-gxx", "g++-13 (Ubuntu 13.4.0) 13.4.0", ("gcc", 13)),
        ("generic-gcc", "gcc (Ubuntu 13.4.0) 13.4.0", ("gcc", 13)),
        ("versioned-gcc", "gcc-13 (Ubuntu 13.4.0) 13.4.0", ("gcc", 13)),
        ("path-versioned-gxx", "/usr/bin/g++-13 (control) 13.2.0", ("gcc", 13)),
        ("punctuated-generic-gxx", "(g++) (control) 13.2.0", ("gcc", 13)),
        ("clang18", "Ubuntu clang version 18.1.3", ("clang", 18)),
        ("generic-gxx-no-version", "g++ (unknown)", ("gcc", None)),
        ("substring-prefix-gxx", "myg++ (control) 13.2.0", (None, None)),
        ("substring-suffix-gxx", "g++driver (control) 13.2.0", (None, None)),
        ("substring-versioned-gxx", "g++-13driver (control) 13.2.0", (None, None)),
        ("substring-prefix-gcc", "mygcc (control) 13.2.0", (None, None)),
        ("substring-suffix-gcc", "gccdriver (control) 13.2.0", (None, None)),
        ("substring-versioned-gcc", "gcc-13driver (control) 13.2.0", (None, None)),
    )
    return [
        (
            "compiler-banner-" + name,
            compiler_version_identity(banner) == expected
            and producer.parse_compiler_version(banner) == expected,
        )
        for name, banner, expected in cases
    ]


def generic_gcc_live(root: Path) -> list[str]:
    with tempfile.TemporaryDirectory(
            prefix="dspark-generic-gcc-live-") as directory:
        record = producer.discover_compiler(
            "gcc13-generic", "gcc", 13, ("g++",), Path(directory))
    errors: list[str] = []
    if record.get("status") != "PASS":
        errors.append("GENERIC_GCC13_DISCOVERY:" + str(record.get("terminal")))
        return errors
    if record.get("candidate") != "g++" \
            or record.get("family") != "gcc" or record.get("major") != 13:
        errors.append("GENERIC_GCC13_IDENTITY")
    first_line = record.get("version_first_line")
    if not isinstance(first_line, str) \
            or compiler_version_identity(first_line) != ("gcc", 13) \
            or producer.parse_compiler_version(first_line) != ("gcc", 13):
        errors.append("GENERIC_GCC13_REAL_BANNER")
    for key in ("version_process", "dump_process"):
        value = record.get(key)
        if not isinstance(value, dict) or value.get("errors") != [] \
                or value.get("timed_out") is not False \
                or value.get("cleanup_residue"):
            errors.append("GENERIC_GCC13_PROCESS:" + key)
    return errors


def self_test(root: Path) -> int:
    baseline = synthetic_transcript()
    source = Path(__file__).read_text(encoding="ascii")
    producer_source = (
        root / "tools/verify_m018_global_corrections.py"
    ).read_text(encoding="ascii")
    threading_source = (
        root / "tools/verify_threading_doc.py"
    ).read_text(encoding="ascii")
    package_outer_errors = package_oracle_outer_binding_errors(source)
    controls: list[tuple[str, bool]] = [
        ("outer-baseline", not validate_transcript(baseline)),
        ("outer-structure", not structural_errors(source, producer_source)),
        ("package-r-meta-outer-structure", not package_outer_errors),
        ("threading-control-structure",
         not threading_control_structure_errors(threading_source)),
    ]
    controls.extend(package_oracle_legal_positive_results(producer_source))
    controls.extend(package_oracle_meta_control_results(source, producer_source))
    controls.extend(process_group_members_race_control())
    ctest_records = [
        {
            "command": list(LITERAL_CTEST_COMMAND),
            "exit_code": 0,
            "timed_out": False,
            "stdout": "PASS global validation contract ctest\n",
            "stderr": "",
            "duration_seconds": 1.0,
            "errors": [],
        }
        for _index in range(REQUIRED_LITERAL_CTEST_REPETITIONS)
    ]
    controls.append((
        "literal-ctest-repetition-baseline",
        not literal_ctest_repetition_errors(ctest_records)))
    controls.append((
        "literal-ctest-repetition-omitted",
        "LITERAL_CTEST_REPEAT_CARDINALITY:23"
        in literal_ctest_repetition_errors(ctest_records[:-1])))

    def catches(name: str, value: dict[str, object], prefix: str,
                **options: object) -> None:
        errors = validate_transcript(value, **options)
        controls.append((name, any(error.startswith(prefix) for error in errors)))

    mutant = copy.deepcopy(baseline)
    mutant["invocation_count"] = 0
    catches("whole-producer-invocation", mutant,
            "VALIDATION_OUTER_MATRIX_INVOCATION_COUNT")
    errors = validate_transcript(
        baseline, expected_ids=EXPECTED_EXECUTION_IDS[:-1])
    controls.append((
        "literal-expected-inventory",
        any(error.startswith(
            "VALIDATION_OUTER_EXPECTED_INVENTORY_CARDINALITY")
            for error in errors)))
    controls.append((
        "inventory-comparison",
        "VALIDATION_OUTER_INVENTORY_COMPARISON_DISABLED"
        in validate_transcript(baseline, comparison_enabled=False)))
    controls.append((
        "per-row-oracle",
        "VALIDATION_OUTER_ROW_ORACLE_DISABLED"
        in validate_transcript(baseline, row_oracle_enabled=False)))
    controls.append((
        "literal-inventory-source",
        "VALIDATION_OUTER_EXPECTED_INVENTORY_LITERAL"
        in structural_errors(source.replace(
            '    "reverb::gcc13::normal::baseline",\n', "", 1),
            producer_source)))
    controls.append((
        "inventory-comparison-source",
        "VALIDATION_OUTER_INVENTORY_COMPARISON_STRUCTURE"
        in structural_errors(source.replace(
            "    elif set(identities) != set(expected_ids):\n",
            "    elif False:\n", 1), producer_source)))
    controls.append((
        "row-oracle-source",
        "VALIDATION_OUTER_ROW_ORACLE_STRUCTURE"
        in structural_errors(source.replace(
            "    if row_oracle_enabled:\n", "    if False:\n", 1),
            producer_source)))
    for identity in EXPECTED_EXECUTION_IDS:
        mutant = copy.deepcopy(baseline)
        mutant["executions"] = [
            row for row in mutant["executions"] if row["id"] != identity]
        catches("identity-" + identity, mutant,
                "VALIDATION_OUTER_MATRIX_IDENTITY_SET")
    for dimension, values in (
        ("compiler", ("gcc13", "clang18")),
        ("profile", ("normal", "sanitizer")),
        ("variant", tuple(EXPECTED_SOURCE_HASHES)),
    ):
        for value in values:
            mutant = copy.deepcopy(baseline)
            mutant["executions"] = [
                row for row in mutant["executions"]
                if row[dimension] != value]
            catches("dimension-{}-{}".format(dimension, value), mutant,
                    "VALIDATION_OUTER_MATRIX_IDENTITY_SET")
    for variant in tuple(EXPECTED_SOURCE_HASHES)[1:]:
        mutant = copy.deepcopy(baseline)
        for source_record in mutant["source_variants"]:
            if source_record["id"] == variant:
                source_record["source_sha256"] = EXPECTED_SOURCE_HASHES["baseline"]
        catches("replacement-" + variant, mutant,
                "VALIDATION_OUTER_SOURCE_HASH")
    mutant = copy.deepcopy(baseline)
    next(row for row in mutant["executions"]
         if row["profile"] == "sanitizer")["oracle_satisfied"] = False
    catches("sanitizer-only-oracle", mutant,
            "VALIDATION_OUTER_ROW_ORACLE")
    mutant = copy.deepcopy(baseline)
    mutant["executions"][0]["run"]["stdout"] += "unexpected\n"
    catches("raw-row-output", mutant,
            "VALIDATION_OUTER_RUN:")
    mutant = copy.deepcopy(baseline)
    mutant["executions"][0]["environment_contract"]["set"]["LANG"] = ""
    catches("runtime-environment", mutant,
            "VALIDATION_OUTER_RUNTIME_ENVIRONMENT")
    controls.append((
        "producer-nonzero",
        "VALIDATION_OUTER_PRODUCER_EXIT:9"
        in validate_transcript(baseline, producer_exit=9)))
    mutant = copy.deepcopy(baseline)
    mutant["executions"][0]["cleanup_residue"] = [999]
    catches("leftover-process", mutant,
            "VALIDATION_OUTER_CLEANUP_RESIDUE")
    mutant = copy.deepcopy(baseline)
    mutant["executions"].append(copy.deepcopy(mutant["executions"][0]))
    catches("duplicate-identity", mutant,
            "VALIDATION_OUTER_OBSERVED_IDENTITY_DUPLICATE")
    mutant = copy.deepcopy(baseline)
    extra = copy.deepcopy(mutant["executions"][0])
    extra["id"] = "reverb::extra"
    mutant["executions"].append(extra)
    catches("extra-identity", mutant,
            "VALIDATION_OUTER_MATRIX_IDENTITY_SET")

    controls.append((
        "delete-live-producer-call",
        "VALIDATION_OUTER_PRODUCER_CALL_CARDINALITY"
        in structural_errors(
            source.replace(LIVE_PRODUCER_CALL, "", 1), producer_source)))
    controls.append((
        "delete-process-group-race-control-call",
        "PROCESS_GROUP_RACE_CONTROL_INVOCATION"
        in structural_errors(source.replace(
            PROCESS_GROUP_RACE_CONTROL_CALL, "", 1), producer_source)))
    controls.append((
        "neutralize-process-group-forced-exception",
        "PROCESS_GROUP_RACE_FORCED_EXCEPTION"
        in structural_errors(source.replace(
            "                raise ProcessLookupError("
            "\"controlled vanished /proc entry\")\n",
            "                return \"41001 (neutralized) S 1 1 1 0\\\\n\"\n",
            1), producer_source)))
    controls.append((
        "remove-process-group-exception-clause",
        "PROCESS_GROUP_RACE_PRODUCER_EXCEPTION_CLAUSE"
        in structural_errors(
            source,
            producer_source.replace(
                "        except (FileNotFoundError, ProcessLookupError, "
                "PermissionError,\n"
                "                ValueError, IndexError):\n",
                "        except (FileNotFoundError, PermissionError,\n"
                "                ValueError, IndexError):\n",
                1))))
    controls.append((
        "accept-wrong-process-group-members",
        "PROCESS_GROUP_RACE_EXACT_MEMBERSHIP_ORACLE"
        in structural_errors(source.replace(
            "    exact_members = members == [valid_pid]\n",
            "    exact_members = members == members\n", 1),
            producer_source)))
    controls.append((
        "delete-producer-matrix-call",
        "VALIDATION_OUTER_PRODUCER_MATRIX_CALL_CARDINALITY"
        in structural_errors(
            source, producer_source.replace(PRODUCER_MATRIX_CALL, "", 1))))
    controls.append((
        "delete-timestretch-rollback-call",
        "VALIDATION_OUTER_TIMESTRETCH_ROLLBACK_CALL"
        in structural_errors(source.replace(
            "                    errors.extend(validate_timestretch_rollback(\n"
            "                        failure_root, before, result, phase, output_name))\n",
            "", 1), producer_source)))
    controls.append((
        "neutralize-timestretch-sentinel-oracle",
        "VALIDATION_OUTER_TIMESTRETCH_SENTINEL_ORACLE"
        in structural_errors(source.replace(
            "    if after != before:\n", "    if False:\n", 1),
            producer_source)))
    controls.append((
        "neutralize-timestretch-cleanup-oracle",
        "VALIDATION_OUTER_TIMESTRETCH_CLEANUP_ORACLE"
        in structural_errors(source.replace(
            "    if timestretch_transaction_residue(directory):\n",
            "    if False:\n", 1), producer_source)))
    controls.append((
        "delete-timestretch-position-loop",
        "VALIDATION_OUTER_TIMESTRETCH_POSITION_LOOPS"
        in structural_errors(source.replace(
            "                for index, output_name in enumerate(EXPECTED_OUTPUTS):\n",
            "                for index, output_name in ():\n", 1),
            producer_source)))
    controls.append((
        "delete-threading-internal-controls",
        "THREADING_INTERNAL_CONTROL_INVOCATION"
        in threading_control_structure_errors(threading_source.replace(
            "build_census_controls(doc, pin_census)", "([], [])", 1))))
    controls.append((
        "threading-external-missing-row",
        "THREADING_EXTERNAL_CARDINALITY:32"
        in threading_external_inventory_errors(
            list(EXPECTED_THREADING_EXTERNAL_IDS[:-1]))))
    workflow_paths = (
        ".github/workflows/ci.yml",
        ".github/workflows/docs.yml",
        "tests/CMakeLists.txt",
    )
    for path in workflow_paths:
        original = (root / path).read_text(encoding="ascii")
        if path.endswith("CMakeLists.txt"):
            changed = original.replace(
                "${PROJECT_SOURCE_DIR}/tools/verify_global_validation_contract.py",
                "${PROJECT_SOURCE_DIR}/tools/verify_m018_global_corrections.py",
                1)
        else:
            changed = original.replace(LIVE_COMMAND, "", 1)
        controls.append((
            "binding-" + path,
            bool(binding_errors(root, {path: changed}))))
    controls.extend(compiler_banner_controls())
    controls.extend(compiler_process_controls())
    for error in package_outer_errors:
        print("ERROR " + error, file=sys.stderr)
    for name, passed in controls:
        print("{} outer mutant {}".format("PASS" if passed else "FAIL", name))
    return 0 if all(passed for _name, passed in controls) else 1


def normal_gate(root: Path) -> dict[str, object]:
    return run_child(
        [sys.executable, "-B", "tools/verify_m018_global_corrections.py"],
        root, 300)


def ctest_mode(root: Path) -> int:
    errors = structural_errors(Path(__file__).read_text(encoding="ascii"))
    errors.extend(threading_control_structure_errors(
        (root / "tools/verify_threading_doc.py").read_text(encoding="ascii")))
    errors.extend(binding_errors(root))
    if self_test(root) != 0:
        errors.append("VALIDATION_OUTER_META_CONTROLS")
    child = normal_gate(root)
    if child["exit_code"] != 0 or child["errors"]:
        errors.append("VALIDATION_OUTER_NORMAL_GATE\n" + str(child["stdout"])
                      + str(child["stderr"]))
    for error in errors:
        print("ERROR " + error, file=sys.stderr)
    if errors:
        return 1
    print("PASS global validation contract ctest")
    return 0


def run_live_producer(root: Path, transcript_path: Path) -> dict[str, object]:
    return run_child(
        [
            sys.executable, "-B",
            "tools/verify_m018_global_corrections.py", "--self-test",
            "--machine-json", str(transcript_path),
        ],
        root, 1200)


def csv_rows(path: Path) -> list[dict[str, str]]:
    lines = [line for line in path.read_text(encoding="ascii").splitlines()
             if line and not line.startswith("#")]
    return list(csv.DictReader(lines))


def measurement_errors(directory: Path) -> list[str]:
    errors: list[str] = []
    expected_counts = {
        "c1-stationary-fidelity.csv": 7,
        "c2-spurious-floor.csv": 7,
        "c3-transient-preservation.csv": 7,
        "c4-vertical-coherence.csv": 14,
        "c5-stereo-integrity.csv": 7,
        "c6-exact-ratio-drift.csv": 7,
        "c7-unity-passthrough.csv": 4,
        "c8-chopping-determinism.csv": 16,
    }
    expected_columns = {
        "c1-stationary-fidelity.csv": (
            "ratio", "lsd_db", "lsd_limit_db", "lsd_db_unclipped",
            "spectral_convergence_db"),
        "c2-spurious-floor.csv": (
            "ratio", "worst_spurious_db_re_fundamental", "worst_bin",
            "worst_hz"),
        "c3-transient-preservation.csv": (
            "ratio", "onsets", "mean_energy_ratio_vs_wsola",
            "worst_energy_ratio_vs_wsola", "source_attack_ms",
            "output_attack_ms", "attack_expansion_percent"),
        "c4-vertical-coherence.csv": (
            "ratio", "bed", "consistency_in", "consistency_out_locked",
            "consistency_out_plain", "ratio_out_over_in", "vcoh_in",
            "vcoh_locked", "vcoh_plain", "locked_better_than_plain"),
        "c5-stereo-integrity.csv": (
            "ratio", "ild_in_db", "ild_out_db", "ild_delta_db",
            "coherence_in", "coherence_out"),
        "c6-exact-ratio-drift.csv": (
            "target_ratio", "input_samples", "output_samples",
            "expected_samples", "length_error_samples", "one_hop",
            "realised_ratio_from_onsets", "ratio_error_percent",
            "onsets_used", "worst_onset_error_samples"),
        "c7-unity-passthrough.csv": (
            "signal", "path", "residual_dbfs", "thd_n_dbfs", "peak_dbfs"),
        "c8-chopping-determinism.csv": (
            "ratio", "pattern", "samples_compared", "differing_samples",
            "first_divergence", "max_abs_difference"),
    }
    parsed: dict[str, list[dict[str, str]]] = {}
    for name, count in expected_counts.items():
        try:
            parsed[name] = csv_rows(directory / name)
        except (OSError, csv.Error, UnicodeError) as error:
            errors.append("TIMESTRETCH_SCHEMA:{}:{}".format(name, error))
            continue
        if len(parsed[name]) != count:
            errors.append("TIMESTRETCH_ROW_COUNT:{}:{}".format(
                name, len(parsed[name])))
        if parsed[name] and tuple(parsed[name][0]) != expected_columns[name]:
            errors.append("TIMESTRETCH_SCHEMA_COLUMNS:" + name)
    try:
        for row in parsed["c1-stationary-fidelity.csv"]:
            if float(row["lsd_db"]) > float(row["lsd_limit_db"]):
                errors.append("TIMESTRETCH_STATIONARY_FIDELITY")
        for row in parsed["c2-spurious-floor.csv"]:
            if float(row["worst_spurious_db_re_fundamental"]) > -80.0:
                errors.append("TIMESTRETCH_SPURIOUS_FLOOR")
        for row in parsed["c3-transient-preservation.csv"]:
            if int(row["onsets"]) != 10 \
                    or float(row["worst_energy_ratio_vs_wsola"]) < 0.65:
                errors.append("TIMESTRETCH_TRANSIENT_PRESERVATION")
        for row in parsed["c4-vertical-coherence.csv"]:
            numeric = tuple(float(row[name]) for name in (
                "ratio", "consistency_in", "consistency_out_locked",
                "consistency_out_plain", "ratio_out_over_in", "vcoh_in",
                "vcoh_locked", "vcoh_plain"))
            if not all(math.isfinite(value) for value in numeric) \
                    or float(row["vcoh_locked"]) <= 0.98 \
                    or row["bed"] not in {"stationary", "vibrato"} \
                    or row["locked_better_than_plain"] not in {"yes", "NO"}:
                errors.append("TIMESTRETCH_VERTICAL_COHERENCE")
        for row in parsed["c5-stereo-integrity.csv"]:
            if abs(float(row["ild_delta_db"])) >= 0.1 \
                    or float(row["coherence_out"]) <= 0.98:
                errors.append("TIMESTRETCH_STEREO_INTEGRITY")
        for row in parsed["c6-exact-ratio-drift.csv"]:
            if int(row["length_error_samples"]) != 0 \
                    or abs(float(row["ratio_error_percent"])) >= 0.01 \
                    or float(row["worst_onset_error_samples"]) > float(row["one_hop"]):
                errors.append("TIMESTRETCH_RATIO_DRIFT")
        for row in parsed["c7-unity-passthrough.csv"]:
            if float(row["residual_dbfs"]) >= -130.0:
                errors.append("TIMESTRETCH_UNITY_PASSTHROUGH")
        for row in parsed["c8-chopping-determinism.csv"]:
            if int(row["differing_samples"]) != 0 \
                    or int(row["first_divergence"]) != -1 \
                    or float(row["max_abs_difference"]) != 0.0:
                errors.append("TIMESTRETCH_CHOPPING_DETERMINISM")
    except (KeyError, ValueError) as error:
        errors.append("TIMESTRETCH_MEASUREMENT_PARSE:" + str(error))
    summary = (directory / "timestretch-metrics.md").read_text(
        encoding="ascii")
    if summary.count("| r=") != 10 or summary.count("| 120->") != 4:
        errors.append("TIMESTRETCH_SUMMARY_ROWS")
    return sorted(set(errors))


def threading_external_inventory_errors(observed: list[str]) -> list[str]:
    errors: list[str] = []
    if len(observed) != len(EXPECTED_THREADING_EXTERNAL_IDS):
        errors.append("THREADING_EXTERNAL_CARDINALITY:{}".format(
            len(observed)))
    if len(observed) != len(set(observed)):
        errors.append("THREADING_EXTERNAL_DUPLICATE_ID")
    if set(observed) != set(EXPECTED_THREADING_EXTERNAL_IDS):
        errors.append("THREADING_EXTERNAL_IDENTITY_SET")
    return errors


def threading_census_mutations(
    text: str,
) -> list[tuple[str, str, str]]:
    begin = "<!-- THREADING_PIN_CENSUS_BEGIN -->"
    end = "<!-- THREADING_PIN_CENSUS_END -->"
    labels = (
        ("total", "All local pin headers"),
        ("template", "Template-parameter pin headers"),
        ("concrete", "Concrete-word pin headers"),
        ("overlap", "Overlap headers"),
    )

    def row(key: str, label: str, source: str) -> tuple[int, list[str], int]:
        lines = source.splitlines(keepends=True)
        matches = [
            index for index, line in enumerate(lines)
            if line.startswith("- {} (".format(label))
        ]
        if len(matches) != 1:
            raise ValueError("{} row cardinality {}".format(key, len(matches)))
        line = lines[matches[0]]
        count = re.search(r"\((\d+)\)", line)
        members = re.findall(r"`([^`]+\.h)`", line)
        if count is None or int(count.group(1)) != len(members):
            raise ValueError("{} row is not a valid baseline".format(key))
        return matches[0], members, int(count.group(1))

    def replace_line(source: str, index: int, replacement: str) -> str:
        lines = source.splitlines(keepends=True)
        newline = "\n" if lines[index].endswith("\n") else ""
        lines[index] = replacement.rstrip("\n") + newline
        return "".join(lines)

    mutations: list[tuple[str, str, str]] = []
    baseline_rows: dict[str, tuple[str, list[str], int]] = {}
    for key, label in labels:
        index, members, count = row(key, label, text)
        baseline_rows[key] = (label, members, count)
        count_mutant = replace_line(
            text, index,
            text.splitlines()[index].replace(
                "({})".format(count), "({})".format(count - 1), 1))
        mutations.append((
            "count-" + key, count_mutant,
            "DOC_COUNT_MISMATCH:" + key))

    for key, label in labels:
        index, members, _count = row(key, label, text)
        original_line = text.splitlines()[index]
        for member_index, member in enumerate(members, 1):
            needle = "`{}`".format(member)
            if original_line.count(needle) != 1:
                raise ValueError("{} member anchor is not unique".format(key))
            mutant = replace_line(
                text, index,
                original_line.replace(
                    needle, "`Core/AnalogRandom.h`", 1))
            mutations.append((
                "member-{}-{:02d}".format(key, member_index), mutant,
                "SOURCE_SET_MISMATCH:" + key))

    stale = text
    for key in ("total", "concrete"):
        label, _members, _count = baseline_rows[key]
        index, members, count = row(key, label, stale)
        members.remove("Analysis/BeatTracker.h")
        stale = replace_line(
            stale, index,
            "- {} ({}): {}.".format(
                label, count - 1,
                ", ".join("`{}`".format(member) for member in members)))
    mutations.append((
        "stale-10-5-7-2", stale, "SOURCE_SET_MISMATCH:"))
    mutations.append((
        "block-absent", text.replace(begin, "", 1),
        "DOC_CENSUS_BLOCK_CARDINALITY"))
    if text.count(begin) != 1 or text.count(end) != 1:
        raise ValueError("census block baseline cardinality changed")
    body = text.split(begin, 1)[1].split(end, 1)[0]
    mutations.append((
        "block-duplicate", text + "\n" + begin + body + end + "\n",
        "DOC_CENSUS_BLOCK_CARDINALITY"))
    return mutations


def threading_external_live(root: Path) -> list[str]:
    errors: list[str] = []
    baseline_doc = (root / "docs/threading.md").read_text(encoding="ascii")
    try:
        mutations = threading_census_mutations(baseline_doc)
    except (OSError, UnicodeError, ValueError) as error:
        return ["THREADING_EXTERNAL_GENERATOR:" + str(error)]
    generated_ids = [name for name, _text, _terminal in mutations]
    errors.extend(threading_external_inventory_errors(generated_ids))
    if errors:
        return errors

    tracked = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, check=False,
        capture_output=True, timeout=30).stdout.split(b"\0")
    tracked_paths = [item.decode("utf-8") for item in tracked if item]
    if len(tracked_paths) != 487 or len(tracked_paths) != len(set(tracked_paths)):
        return ["THREADING_EXTERNAL_TRACKED_CENSUS:{}".format(
            len(tracked_paths))]

    observed: list[str] = []
    with tempfile.TemporaryDirectory(
            prefix="dspark-threading-external-") as directory:
        scratch = Path(directory)
        base = scratch / "base"
        base.mkdir()
        for relative in tracked_paths:
            source = root / relative
            destination = base / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.is_symlink() or not source.is_file():
                return ["THREADING_EXTERNAL_NONREGULAR_SOURCE:" + relative]
            shutil.copy2(source, destination)

        for name, mutant_doc, expected_terminal in mutations:
            fixture = scratch / name
            shutil.copytree(base, fixture, copy_function=os.link)
            doc_path = fixture / "docs/threading.md"
            doc_path.unlink()
            doc_path.write_text(mutant_doc, encoding="ascii", newline="")
            setup = subprocess.run(
                ["git", "init", "-q"], cwd=fixture, check=False,
                capture_output=True, timeout=30)
            indexed = subprocess.run(
                ["git", "add", "-f", "-A"], cwd=fixture, check=False,
                capture_output=True, timeout=60)
            indexed_paths = subprocess.run(
                ["git", "ls-files", "-z"], cwd=fixture, check=False,
                capture_output=True, timeout=30)
            indexed_count = len([
                item for item in indexed_paths.stdout.split(b"\0") if item
            ])
            if setup.returncode != 0 or indexed.returncode != 0 \
                    or indexed_paths.returncode != 0 or indexed_count != 487:
                errors.append("THREADING_EXTERNAL_FIXTURE:{}".format(name))
                observed.append(name)
                continue
            result = producer.run_owned(
                [sys.executable, "-B", "tools/verify_threading_doc.py"],
                fixture, 120, "threading_external")
            combined = str(result.get("stdout")) + str(result.get("stderr"))
            prefix = "THREADING_EXTERNAL_MUTANT:" + name
            if result.get("exit_code") == 0 or result.get("errors") != []:
                errors.append(prefix + ":PROCESS")
            if expected_terminal not in combined:
                errors.append(prefix + ":NAMED_TERMINAL")
            if "Traceback" in combined or "RuntimeError" in combined:
                errors.append(prefix + ":UNCAUGHT_EXCEPTION")
            if "SKIP threading census internal controls: external baseline invalid" \
                    not in combined:
                errors.append(prefix + ":BASELINE_FIRST")
            if "threading census mutant" in combined:
                errors.append(prefix + ":LATE_INTERNAL_CONTROL")
            observed.append(name)
    errors.extend(threading_external_inventory_errors(observed))
    return errors


def timestretch_snapshot(directory: Path) -> dict[str, str] | None:
    snapshot: dict[str, str] = {}
    for name in EXPECTED_OUTPUTS:
        path = directory / name
        if path.is_symlink() or not path.is_file():
            return None
        snapshot[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return snapshot


def write_timestretch_sentinels(directory: Path, seed: str) -> dict[str, str]:
    directory.mkdir()
    for index, name in enumerate(EXPECTED_OUTPUTS):
        (directory / name).write_bytes(
            "DSPark transaction sentinel {} {} {}\n".format(
                seed, index, name).encode("ascii"))
    snapshot = timestretch_snapshot(directory)
    if snapshot is None:
        raise RuntimeError("sentinel construction failed")
    return snapshot


def timestretch_transaction_residue(directory: Path) -> list[str]:
    return sorted(
        path.name for path in directory.iterdir()
        if path.name == TIMESTRETCH_TRANSACTION_DIRECTORY
        or path.name.startswith(".dspark-timestretch-transaction-")
    )


def validate_timestretch_rollback(
    directory: Path,
    before: dict[str, str],
    result: dict[str, object],
    phase: str,
    output_name: str,
) -> list[str]:
    errors: list[str] = []
    prefix = "TIMESTRETCH_{}_ROLLBACK:{}".format(phase, output_name)
    expected_stderr = "ERROR TIMESTRETCH_TRANSACTION_{} {}\n".format(
        phase, output_name)
    if result.get("exit_code") != 3 or result.get("errors") != []:
        errors.append(prefix + ":PROCESS")
    if result.get("stderr") != expected_stderr:
        errors.append(prefix + ":TYPED_TERMINAL")
    if "characterisation written" in str(result.get("stdout")):
        errors.append(prefix + ":FALSE_SUCCESS")
    if sorted(path.name for path in directory.iterdir()) \
            != sorted(EXPECTED_OUTPUTS):
        errors.append(prefix + ":FINAL_CENSUS")
    after = timestretch_snapshot(directory)
    if after != before:
        errors.append(prefix + ":SENTINEL_HASH_DRIFT")
    if timestretch_transaction_residue(directory):
        errors.append(prefix + ":TRANSACTION_RESIDUE")
    return errors


def timestretch_tree_snapshot(directory: Path) -> tuple[tuple[object, ...], ...]:
    rows: list[tuple[object, ...]] = []
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory).as_posix()
        mode = stat.S_IMODE(path.lstat().st_mode)
        if path.is_symlink():
            rows.append((relative, "symlink", mode, os.readlink(path)))
        elif path.is_dir():
            rows.append((relative, "directory", mode))
        elif path.is_file():
            data = path.read_bytes()
            rows.append((relative, "regular", mode, len(data),
                         hashlib.sha256(data).hexdigest(), data.hex()))
        else:
            rows.append((relative, "nonregular", mode))
    return tuple(rows)


def write_transaction_fixture(directory: Path, seed: str,
                              present: bool = True) -> dict[str, str]:
    directory.mkdir()
    if present:
        for index, name in enumerate(EXPECTED_OUTPUTS):
            (directory / name).write_bytes(
                "DSPark durable fixture {} {} {}\n".format(
                    seed, index, name).encode("ascii"))
    return timestretch_snapshot(directory) or {}


def transaction_environment(profile: str,
                            values: dict[str, str] | None = None) -> dict[str, str]:
    environment = dict(os.environ)
    for name in REMOVED_RUNTIME_ENVIRONMENT:
        environment.pop(name, None)
    environment.update(NORMAL_RUNTIME_ENVIRONMENT if profile == "normal"
                       else SANITIZER_RUNTIME_ENVIRONMENT)
    for name in (
        "DSPARK_TIMESTRETCH_DURABLE_FAULT",
        "DSPARK_TIMESTRETCH_DURABLE_MODE",
        "DSPARK_TIMESTRETCH_CRASH_CUT",
        "DSPARK_TIMESTRETCH_FAIL_STAGE_INDEX",
        "DSPARK_TIMESTRETCH_FAIL_COMMIT_INDEX",
        "DSPARK_TIMESTRETCH_TRACE_FILE",
        "DSPARK_TIMESTRETCH_PROBE_SEED",
    ):
        environment.pop(name, None)
    if values:
        environment.update(values)
    return environment


def transaction_probe_source(root: Path) -> str:
    source = (root / "tools/characterize_timestretch.cpp").as_posix()
    return r'''#define main dspark_characterizer_entry_not_used
#include "{}"
#undef main

#include <cstdlib>
#include <fstream>
#include <string>

int main(int argc, char** argv)
{{
    if (argc != 3)
        return 90;
    const std::filesystem::path root(argv[1]);
    const std::string operation(argv[2]);
    if (operation != "run" && operation != "recover")
        return 91;
    std::error_code transactionStatusError;
    const bool transactionExists = std::filesystem::exists(
        root / kTransactionDirectory, transactionStatusError);
    if (operation == "recover" && !transactionStatusError
        && !transactionExists)
        return 0;
    OutputTransaction transaction(root);
    if (!transaction.begin())
        return transaction.abortAndReport();
    if (transaction.recoveredCommitted() || operation == "recover")
        return 0;
    const char* selected = std::getenv("DSPARK_TIMESTRETCH_PROBE_SEED");
    const std::string seed = selected == nullptr ? "default" : selected;
    for (size_t index = 0; index < kExpectedOutputs.size(); ++index)
    {{
        std::ofstream output;
        if (!transaction.openOutput(output, kExpectedOutputs[index]))
            return transaction.abortAndReport();
        output << "DSPark durable new " << seed << ' ' << index << ' '
               << kExpectedOutputs[index] << '\n';
        if (!transaction.finishOutput(output, kExpectedOutputs[index]))
            return transaction.abortAndReport();
    }}
    if (!transaction.commit())
        return transaction.reportFailure();
    return 0;
}}
'''.format(source)


def coherence_probe_source(root: Path) -> str:
    source = (root / "tools/characterize_timestretch.cpp").as_posix()
    return r'''#define main dspark_characterizer_entry_not_used
#include "{}"
#undef main

#include <array>
#include <filesystem>
#include <iostream>
#include <string>

int main(int argc, char** argv)
{{
    if (argc < 2)
        return 90;
    const std::string operation(argv[1]);
    if (operation == "pair")
    {{
        if (argc != 3)
            return 91;
        const std::filesystem::path root(argv[2]);
        const std::array<ParsedTransactionJournal, 2> slots = {{
            readTransactionJournal(root / "journal.a"),
            readTransactionJournal(root / "journal.b"),
        }};
        const std::array<ParsedTransactionJournal, 2> temporaries = {{
            readTransactionJournal(root / "journal.a.tmp"),
            readTransactionJournal(root / "journal.b.tmp"),
        }};
        const std::string rootBinding = transactionSha256(
            std::filesystem::absolute(root).lexically_normal().generic_string());
        const std::string transactionId = transactionSha256(
            "DSPark-TimeStretch-transaction-id|" + rootBinding).substr(0, 32);
        int selected = -1;
        const bool accepted = transactionClassifyAuthority(
            slots, temporaries, rootBinding, transactionId, selected);
        if (accepted)
        {{
            std::cout << "ACCEPT\n";
            return 0;
        }}
        std::cerr << "ERROR TIMESTRETCH_TRANSACTION_RECOVERY_COLLISION "
                     ".dspark-timestretch-transaction\n";
        return 6;
    }}
    if (operation == "transition")
    {{
        if (argc != 4)
            return 92;
        const ParsedTransactionJournal low =
            readTransactionJournal(std::filesystem::path(argv[2]));
        const ParsedTransactionJournal high =
            readTransactionJournal(std::filesystem::path(argv[3]));
        std::cout << (transactionValidateAdjacent(low, high)
                      ? "ACCEPT" : "REJECT") << '\n';
        return 0;
    }}
    return 93;
}}
'''.format(source)


def compile_coherence_probes(
    root: Path, scratch: Path, discovery: list[dict[str, object]],
) -> tuple[dict[tuple[str, str], Path], list[str]]:
    source = scratch / "coherence_probe.cpp"
    source.write_text(coherence_probe_source(root), encoding="ascii", newline="")
    compilers = {str(item["label"]): str(item["path"])
                 for item in discovery}
    binaries: dict[tuple[str, str], Path] = {}
    errors: list[str] = []
    for compiler, profile in TIMESTRETCH_BUILD_LANES:
        binary = scratch / "coherence-{}-{}".format(compiler, profile)
        command = [
            compilers[compiler], "-std=c++20", "-O1", "-g", "-Wall",
            "-Wextra", "-Wpedantic", "-Werror", "-I", str(root),
            "-DDSPARK_TIMESTRETCH_TRANSACTION_TESTING=1",
        ]
        if profile == "sanitizer":
            command.extend(SANITIZER_FLAGS)
        command.extend([str(source), "-o", str(binary)])
        result = producer.run_owned(command, scratch, 240, "compile")
        if result["exit_code"] != 0 or result["errors"]:
            errors.append("TIMESTRETCH_COHERENCE_BUILD:{}:{}:{}".format(
                compiler, profile, result.get("stderr")))
        else:
            binaries[(compiler, profile)] = binary
    return binaries, errors


def compile_transaction_probes(
    root: Path, scratch: Path, discovery: list[dict[str, object]],
) -> tuple[dict[tuple[str, str], Path], list[str]]:
    source = scratch / "transaction_probe.cpp"
    source.write_text(transaction_probe_source(root), encoding="ascii", newline="")
    compilers = {str(item["label"]): str(item["path"])
                 for item in discovery}
    binaries: dict[tuple[str, str], Path] = {}
    errors: list[str] = []
    for compiler, profile in TIMESTRETCH_BUILD_LANES:
        binary = scratch / ("transaction-{}-{}".format(compiler, profile))
        command = [
            compilers[compiler], "-std=c++20", "-O1", "-g", "-Wall",
            "-Wextra", "-Wpedantic", "-Werror", "-I", str(root),
            "-DDSPARK_TIMESTRETCH_TRANSACTION_TESTING=1",
        ]
        if profile == "sanitizer":
            command.extend(SANITIZER_FLAGS)
        command.extend([str(source), "-o", str(binary)])
        result = producer.run_owned(command, scratch, 240, "compile")
        if result["exit_code"] != 0 or result["errors"]:
            errors.append("TIMESTRETCH_DURABLE_BUILD:{}:{}:{}".format(
                compiler, profile, result.get("stderr")))
        else:
            binaries[(compiler, profile)] = binary
    return binaries, errors


def run_transaction_probe(binary: Path, output: Path, operation: str,
                          profile: str, values: dict[str, str] | None = None,
                          cwd: Path | None = None) -> dict[str, object]:
    return run_child(
        [str(binary), str(output), operation], cwd or output.parent, 30,
        transaction_environment(profile, values))


def is_recovery_required(result: dict[str, object], kind: str) -> bool:
    code = 4 if kind == "ROLLBACK" else 5
    terminal = "ERROR TIMESTRETCH_TRANSACTION_RECOVERY_REQUIRED_{} ".format(kind)
    return result.get("exit_code") == code and terminal in str(result.get("stderr")) \
        and result.get("errors") == [] \
        and "characterisation written" not in str(result.get("stdout"))


def durable_point_needs_rollback(point: str) -> bool:
    return any(marker in point for marker in (
        "rollback-remove", "rollback-restore", "new-final-remove",
        "restore-rename"))


def durable_point_needs_recovery_seed(point: str) -> bool:
    return "recovery-reconcile" in point


def durable_point_is_postcommit(point: str) -> bool:
    return any(marker in point for marker in (
        "cleanup-backup", "backup-cleanup", "finalize",
        "journal-cleanup", "directory-cleanup"))


def crash_point_needs_rollback_seed(point: str) -> bool:
    return point.startswith("rollback-remove-") \
        or point.startswith("rollback-restore-")


def crash_point_is_postcommit(point: str) -> bool:
    return point == "post-commit-marker" or any(
        point.startswith(prefix) for prefix in (
            "cleanup-backup-", "journal-cleanup-", "directory-cleanup-"))


def expected_probe_snapshot(seed: str) -> dict[str, str]:
    return {
        name: hashlib.sha256(
            "DSPark durable new {} {} {}\n".format(
                seed, index, name).encode("ascii")).hexdigest()
        for index, name in enumerate(EXPECTED_OUTPUTS)
    }


def durable_process_clean(result: dict[str, object]) -> bool:
    combined = str(result.get("stdout")) + str(result.get("stderr"))
    return result.get("errors") == [] \
        and not any(item in combined for item in SANITIZER_DIAGNOSTICS) \
        and "characterisation written" not in str(result.get("stdout"))


def old_authority_retained(directory: Path,
                           expected: dict[str, str]) -> bool:
    backup = directory / TIMESTRETCH_TRANSACTION_DIRECTORY / "backup"
    for name, digest in expected.items():
        matches = 0
        for path in (directory / name, backup / name):
            if path.is_symlink():
                return False
            if path.is_file():
                observed = hashlib.sha256(path.read_bytes()).hexdigest()
                if observed == digest:
                    matches += 1
                elif path.parent == backup:
                    return False
        if matches != 1:
            return False
    return True


def rewrite_transaction_checksum(bytes_value: bytes) -> bytes:
    marker = b"payload_sha256="
    offset = bytes_value.rfind(marker)
    if offset < 0:
        raise ValueError("journal checksum field missing")
    payload = bytes_value[:offset]
    return payload + marker + hashlib.sha256(payload).hexdigest().encode("ascii") + b"\n"


COHERENCE_ZERO_SHA256 = "0" * 64
COHERENCE_MAX_GENERATION = (1 << 64) - 1


def coherence_sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def coherence_entry(index: int, established: bool = False) \
        -> dict[str, object]:
    return {
        "output_index": index,
        "expected_name": EXPECTED_OUTPUTS[index],
        "old_present": True,
        "old_size": 1000 + index,
        "old_sha256": coherence_sha("old::{}".format(index)),
        "old_location": "FINAL",
        "new_size": 2000 + index if established else 0,
        "new_sha256": coherence_sha("new::{}".format(index))
        if established else COHERENCE_ZERO_SHA256,
        "new_location": "STAGED" if established else "NONE",
    }


def coherence_record(
    tag: str,
    state: str,
    root_binding: str,
    transaction_id: str,
    *,
    established: int = 0,
    generation: int = 10,
    pending_kind: str = "NONE",
    pending_index: int = 9,
    rollback_cursor: int = 0,
    cleanup_cursor: int = 0,
    finalize_cursor: int = 0,
) -> dict[str, object]:
    return {
        "transaction_id": transaction_id,
        "root_binding_sha256": root_binding,
        "generation": generation,
        "previous_payload_sha256": coherence_sha("prior::" + tag),
        "transition_id": "T01",
        "transition_index": generation - 1,
        "state": state,
        "pending_kind": pending_kind,
        "pending_index": pending_index,
        "rollback_cursor": rollback_cursor,
        "cleanup_cursor": cleanup_cursor,
        "finalize_cursor": finalize_cursor,
        "outputs": [coherence_entry(index, index < established)
                    for index in range(9)],
    }


def coherence_set_all_old(record: dict[str, object], location: str) -> None:
    for output in record["outputs"]:
        output["old_location"] = location


def coherence_set_all_new(record: dict[str, object], location: str) -> None:
    for output in record["outputs"]:
        if output["new_size"]:
            output["new_location"] = location


def coherence_payload(record: dict[str, object]) -> bytes:
    lines = [
        "magic=DSPARK_TIMESTRETCH_TXN",
        "version=2",
        "transaction_id=" + str(record["transaction_id"]),
        "root_binding_sha256=" + str(record["root_binding_sha256"]),
        "generation=" + str(record["generation"]),
        "previous_payload_sha256="
        + str(record["previous_payload_sha256"]),
        "transition_id=" + str(record["transition_id"]),
        "transition_index=" + str(record["transition_index"]),
        "state=" + str(record["state"]),
        "pending_kind=" + str(record["pending_kind"]),
        "pending_index=" + str(record["pending_index"]),
        "rollback_cursor=" + str(record["rollback_cursor"]),
        "cleanup_cursor=" + str(record["cleanup_cursor"]),
        "finalize_cursor=" + str(record["finalize_cursor"]),
    ]
    for index, output in enumerate(record["outputs"]):
        lines.append("output{}={}|{}|{}|{}|{}|{}|{}|{}|{}".format(
            index,
            output["output_index"], output["expected_name"],
            1 if output["old_present"] else 0,
            output["old_size"], output["old_sha256"],
            output["old_location"], output["new_size"],
            output["new_sha256"], output["new_location"]))
    return ("\n".join(lines) + "\n").encode("ascii")


def coherence_serialize(record: dict[str, object]) -> bytes:
    payload = coherence_payload(record)
    return payload + (
        "payload_sha256=" + hashlib.sha256(payload).hexdigest() + "\n"
    ).encode("ascii")


def coherence_apply(
    low: dict[str, object], transition_id: str,
) -> dict[str, object]:
    high = copy.deepcopy(low)
    high["state"] = next(
        row[2] for row in TIMESTRETCH_COHERENCE_TRANSITIONS
        if row[0] == transition_id)
    outputs = high["outputs"]
    low_outputs = low["outputs"]
    if transition_id == "T02":
        outputs[0].update(coherence_entry(0, True))
    elif transition_id in ("T03", "T04"):
        target = next(index for index, output in enumerate(outputs)
                      if output["new_size"] == 0)
        outputs[target].update(coherence_entry(target, True))
    elif transition_id == "T05":
        high["pending_kind"], high["pending_index"] = "BACKUP", 0
    elif transition_id == "T06":
        target = int(low["pending_index"])
        outputs[target]["old_location"] = (
            "BACKUP" if outputs[target]["old_present"] else "ABSENT")
        high["pending_kind"], high["pending_index"] = "NONE", 9
    elif transition_id == "T07":
        target = next(index for index, output in enumerate(low_outputs)
                      if output["old_present"]
                      and output["old_location"] == "FINAL")
        high["pending_kind"], high["pending_index"] = "BACKUP", target
    elif transition_id == "T08":
        high["pending_kind"], high["pending_index"] = "PUBLISH", 0
    elif transition_id == "T09":
        target = int(low["pending_index"])
        outputs[target]["new_location"] = "FINAL"
        high["pending_kind"], high["pending_index"] = "NONE", 9
    elif transition_id == "T10":
        target = next(index for index, output in enumerate(low_outputs)
                      if output["new_location"] == "STAGED")
        high["pending_kind"], high["pending_index"] = "PUBLISH", target
    elif transition_id in tuple("T{:02d}".format(index)
                                for index in range(13, 19)):
        high["rollback_cursor"] = 0
        high["cleanup_cursor"] = 0
        high["finalize_cursor"] = 0
    elif transition_id == "T19":
        high["pending_kind"] = "REMOVE_NEW"
        high["pending_index"] = 8 - int(low["rollback_cursor"])
    elif transition_id == "T20":
        target = int(low["pending_index"])
        outputs[target]["new_location"] = "NONE"
        high["pending_kind"], high["pending_index"] = "NONE", 9
        high["rollback_cursor"] = int(low["rollback_cursor"]) + 1
    elif transition_id == "T21":
        high["pending_kind"] = "RESTORE_OLD"
        high["pending_index"] = 17 - int(low["rollback_cursor"])
    elif transition_id == "T22":
        target = int(low["pending_index"])
        outputs[target]["old_location"] = (
            "FINAL" if outputs[target]["old_present"] else "ABSENT")
        high["pending_kind"], high["pending_index"] = "NONE", 9
        high["rollback_cursor"] = int(low["rollback_cursor"]) + 1
    elif transition_id == "T26":
        high["finalize_cursor"] = int(low["finalize_cursor"]) + 1
    elif transition_id == "T27":
        high["pending_kind"] = "CLEANUP_BACKUP"
        high["pending_index"] = int(low["cleanup_cursor"])
    elif transition_id == "T28":
        target = int(low["pending_index"])
        outputs[target]["old_location"] = "ABSENT"
        high["pending_kind"], high["pending_index"] = "NONE", 9
        high["cleanup_cursor"] = int(low["cleanup_cursor"]) + 1
    elif transition_id == "T32":
        high["finalize_cursor"] = int(low["finalize_cursor"]) + 1
    high["generation"] = int(low["generation"]) + 1
    high["previous_payload_sha256"] = hashlib.sha256(
        coherence_payload(low)).hexdigest()
    high["transition_id"] = transition_id
    high["transition_index"] = int(low["transition_index"]) + 1
    return high


def coherence_fixture(
    transition_id: str, root_binding: str, transaction_id: str,
) -> dict[str, object]:
    make = lambda state, **values: coherence_record(  # noqa: E731
        transition_id, state, root_binding, transaction_id, **values)
    if transition_id in ("T01", "T02", "T13"):
        return make("SETUP")
    if transition_id == "T03":
        return make("STAGING", established=4)
    if transition_id in ("T04", "T14"):
        return make("STAGING", established=8)
    if transition_id in ("T05", "T15"):
        return make("STAGED", established=9)
    if transition_id == "T06":
        return make("BACKUP_IN_PROGRESS", established=9,
                    pending_kind="BACKUP", pending_index=0)
    if transition_id == "T07":
        record = make("BACKUP_IN_PROGRESS", established=9)
        record["outputs"][0]["old_location"] = "BACKUP"
        return record
    if transition_id == "T08":
        record = make("BACKUP_IN_PROGRESS", established=9)
        coherence_set_all_old(record, "BACKUP")
        return record
    if transition_id == "T09":
        record = make("PUBLISH_IN_PROGRESS", established=9,
                      pending_kind="PUBLISH", pending_index=0)
        coherence_set_all_old(record, "BACKUP")
        return record
    if transition_id == "T10":
        record = make("PUBLISH_IN_PROGRESS", established=9)
        coherence_set_all_old(record, "BACKUP")
        record["outputs"][0]["new_location"] = "FINAL"
        return record
    if transition_id in ("T11", "T17"):
        record = make("PUBLISH_IN_PROGRESS", established=9)
        coherence_set_all_old(record, "BACKUP")
        coherence_set_all_new(record, "FINAL")
        return record
    if transition_id in ("T12", "T18"):
        record = make("COMMIT_READY", established=9)
        coherence_set_all_old(record, "BACKUP")
        coherence_set_all_new(record, "FINAL")
        return record
    if transition_id == "T16":
        record = make("BACKUP_IN_PROGRESS", established=9)
        record["outputs"][0]["old_location"] = "BACKUP"
        return record
    if transition_id == "T19":
        record = make("ROLLBACK_REQUIRED", established=9)
        coherence_set_all_old(record, "BACKUP")
        coherence_set_all_new(record, "FINAL")
        return record
    if transition_id == "T20":
        record = coherence_fixture("T19", root_binding, transaction_id)
        record["previous_payload_sha256"] = coherence_sha("prior::T20")
        record["pending_kind"], record["pending_index"] = "REMOVE_NEW", 8
        return record
    if transition_id == "T21":
        record = make("ROLLBACK_REQUIRED", established=9,
                      rollback_cursor=9)
        coherence_set_all_old(record, "BACKUP")
        coherence_set_all_new(record, "NONE")
        return record
    if transition_id == "T22":
        record = coherence_fixture("T21", root_binding, transaction_id)
        record["previous_payload_sha256"] = coherence_sha("prior::T22")
        record["pending_kind"], record["pending_index"] = "RESTORE_OLD", 8
        return record
    if transition_id == "T23":
        record = coherence_fixture("T20", root_binding, transaction_id)
        record["previous_payload_sha256"] = coherence_sha("prior::T23")
        return record
    if transition_id in ("T24", "T25"):
        record = coherence_fixture("T23", root_binding, transaction_id)
        record["previous_payload_sha256"] = coherence_sha(
            "prior::" + transition_id)
        record["state"] = "RECOVERY_REQUIRED_ROLLBACK"
        return record
    if transition_id == "T26":
        record = make("ROLLBACK_REQUIRED", established=9,
                      rollback_cursor=18)
        coherence_set_all_new(record, "NONE")
        coherence_set_all_old(record, "FINAL")
        return record
    if transition_id == "T27":
        record = make("COMMITTED_CLEANUP_PENDING", established=9)
        coherence_set_all_old(record, "BACKUP")
        coherence_set_all_new(record, "FINAL")
        return record
    if transition_id == "T28":
        record = coherence_fixture("T27", root_binding, transaction_id)
        record["previous_payload_sha256"] = coherence_sha("prior::T28")
        record["pending_kind"], record["pending_index"] = (
            "CLEANUP_BACKUP", 0)
        return record
    if transition_id == "T29":
        record = coherence_fixture("T28", root_binding, transaction_id)
        record["previous_payload_sha256"] = coherence_sha("prior::T29")
        return record
    if transition_id in ("T30", "T31"):
        record = coherence_fixture("T29", root_binding, transaction_id)
        record["previous_payload_sha256"] = coherence_sha(
            "prior::" + transition_id)
        record["state"] = "RECOVERY_REQUIRED_CLEANUP"
        return record
    if transition_id == "T32":
        record = make("COMMITTED_CLEANUP_PENDING", established=9,
                      cleanup_cursor=9)
        coherence_set_all_old(record, "ABSENT")
        coherence_set_all_new(record, "FINAL")
        return record
    raise ValueError("unknown coherence fixture " + transition_id)


def coherence_root_identity(root: Path) -> tuple[str, str]:
    root_binding = hashlib.sha256(
        root.absolute().as_posix().encode("ascii")).hexdigest()
    transaction_id = hashlib.sha256(
        ("DSPark-TimeStretch-transaction-id|" + root_binding)
        .encode("ascii")).hexdigest()[:32]
    return root_binding, transaction_id


def coherence_pair_fixture(
    slug: str, root_binding: str, transaction_id: str,
) -> tuple[list[bytes | None], list[bytes | None], str]:
    fixtures = {
        transition_id: (
            coherence_fixture(transition_id, root_binding, transaction_id),
            None,
        )
        for transition_id, _source, _target, _event
        in TIMESTRETCH_COHERENCE_TRANSITIONS
    }
    for transition_id, pair in fixtures.items():
        pair = (pair[0], coherence_apply(pair[0], transition_id))
        fixtures[transition_id] = pair
    t01_low, t01_high = fixtures["T01"]
    _t12_low, t12_high = fixtures["T12"]
    t13_low, t13_high = fixtures["T13"]
    t24_low, t24_high = fixtures["T24"]
    t30_low, t30_high = fixtures["T30"]
    _t27_low, t27_high = fixtures["T27"]

    def altered(record: dict[str, object], **values: object) \
            -> dict[str, object]:
        result = copy.deepcopy(record)
        result.update(values)
        return result

    genesis = altered(
        t01_low, generation=1, transition_index=0,
        previous_payload_sha256=COHERENCE_ZERO_SHA256,
        transition_id="BEGIN")
    max_low = altered(
        t01_low, generation=COHERENCE_MAX_GENERATION - 1,
        transition_index=COHERENCE_MAX_GENERATION - 2)
    max_high = coherence_apply(max_low, "T01")
    zero = altered(t01_low, generation=0, transition_index=0)
    bad_predecessor = altered(
        t01_high, previous_payload_sha256=coherence_sha("wrong"))
    foreign_transaction = altered(
        t01_high, transaction_id=coherence_sha("foreign")[:32])
    foreign_root = altered(
        t01_high, root_binding_sha256=coherence_sha("foreign-root"))
    gap = altered(
        t01_high, generation=int(t01_low["generation"]) + 2,
        transition_index=int(t01_low["transition_index"]) + 2)
    equal_different = altered(
        t01_low, previous_payload_sha256=coherence_sha("equal-different"))
    committed_to_setup = altered(
        t12_high,
        generation=int(t12_high["generation"]) + 1,
        transition_index=int(t12_high["transition_index"]) + 1,
        previous_payload_sha256=hashlib.sha256(
            coherence_payload(t12_high)).hexdigest(),
        transition_id="T01", state="SETUP")

    specifications: dict[str, tuple[list[object | None],
                                    list[object | None], str]] = {
        "legal-single-genesis": ([genesis, None], [None, None], "ACCEPT"),
        "legal-single-precommit": ([t01_low, None], [None, None], "ACCEPT"),
        "legal-single-committed": ([t12_high, None], [None, None], "ACCEPT"),
        "legal-linked-precommit": ([t01_low, t01_high], [None, None], "ACCEPT"),
        "legal-linked-enter-rollback": ([t13_low, t13_high], [None, None], "ACCEPT"),
        "legal-linked-committed-cleanup": (list(fixtures["T27"]), [None, None], "ACCEPT"),
        "legal-equal-identical": ([t01_low, copy.deepcopy(t01_low)], [None, None], "ACCEPT"),
        "legal-stale-lower": ([t01_high, t01_low], [None, None], "ACCEPT"),
        "legal-idempotent-rollback-retry": ([t24_low, t24_high], [None, None], "ACCEPT"),
        "legal-idempotent-cleanup-retry": ([t30_low, t30_high], [None, None], "ACCEPT"),
        "predecessor-mismatch": ([t01_low, bad_predecessor], [None, None], "COLLISION"),
        "replay-nonpredecessor": ([fixtures["T02"][0], t01_high], [None, None], "COLLISION"),
        "foreign-transaction": ([t01_low, foreign_transaction], [None, None], "COLLISION"),
        "foreign-root": ([t01_low, foreign_root], [None, None], "COLLISION"),
        "generation-gap": ([t01_low, gap], [None, None], "COLLISION"),
        "equal-different": ([t01_low, equal_different], [None, None], "COLLISION"),
        "lower-generation-zero": ([zero, t01_high], [None, None], "COLLISION"),
        "higher-generation-zero": ([t01_low, zero], [None, None], "COLLISION"),
        "committed-to-setup": ([t12_high, committed_to_setup], [None, None], "COLLISION"),
        "generation-max-single": ([max_high, None], [None, None], "COLLISION"),
        "generation-max-pair": ([max_low, max_high], [None, None], "COLLISION"),
        "valid-plus-invalid-final": ([t01_low, b"invalid\n"], [None, None], "COLLISION"),
        "legacy-v1-single": (["LEGACY", None], [None, None], "COLLISION"),
        "legacy-v1-pair": (["LEGACY", "LEGACY"], [None, None], "COLLISION"),
        "mixed-v1-v2": (["LEGACY", t01_high], [None, None], "COLLISION"),
        "linked-temp-unpromoted": ([t01_low, None], [None, t01_high], "ACCEPT"),
        "bad-temp-predecessor": ([t01_low, None], [None, bad_predecessor], "COLLISION"),
        "two-finals-plus-temp": ([t01_low, t01_high], [None, t27_high], "COLLISION"),
        "no-valid-slot": ([None, None], [None, None], "COLLISION"),
    }
    for state, state_slug in (
        ("STAGING", "committed-to-staging"),
        ("STAGED", "committed-to-staged"),
        ("BACKUP_IN_PROGRESS", "committed-to-backup"),
        ("PUBLISH_IN_PROGRESS", "committed-to-publish"),
        ("COMMIT_READY", "committed-to-commit-ready"),
        ("ROLLBACK_REQUIRED", "committed-to-rollback"),
        ("RECOVERY_REQUIRED_ROLLBACK", "committed-to-recovery-rollback"),
    ):
        specifications[state_slug] = (
            [t12_high, altered(committed_to_setup, state=state)],
            [None, None], "COLLISION")
    finals, temporaries, expected = specifications[slug]

    def serialized(value: object | None) -> bytes | None:
        if value is None or isinstance(value, bytes):
            return value
        if value == "LEGACY":
            legacy = coherence_serialize(t01_low).replace(
                b"version=2\n", b"version=1\n", 1)
            return rewrite_transaction_checksum(legacy)
        return coherence_serialize(value)

    return ([serialized(value) for value in finals],
            [serialized(value) for value in temporaries], expected)


def coherence_metadata_fixture(
    slug: str, root_binding: str, transaction_id: str,
) -> tuple[bytes, bytes]:
    low = coherence_fixture("T05", root_binding, transaction_id)
    high = coherence_apply(low, "T05")
    mutant = copy.deepcopy(high)
    if slug == "transaction-id":
        mutant["transaction_id"] = coherence_sha("mutant")[:32]
    elif slug == "root-binding":
        mutant["root_binding_sha256"] = coherence_sha("mutant-root")
    elif slug == "output-index":
        mutant["outputs"][0]["output_index"] = 1
    elif slug == "expected-name":
        mutant["outputs"][0]["expected_name"] = "wrong.csv"
    elif slug == "old-present":
        mutant["outputs"][0]["old_present"] = False
    elif slug == "old-size":
        mutant["outputs"][0]["old_size"] = 777
    elif slug == "old-sha256":
        mutant["outputs"][0]["old_sha256"] = coherence_sha("wrong-old")
    elif slug == "new-size-established":
        mutant["outputs"][0]["new_size"] = 777
    elif slug == "new-sha256-established":
        mutant["outputs"][0]["new_sha256"] = coherence_sha("wrong-new")
    elif slug == "old-location":
        mutant["outputs"][0]["old_location"] = "BACKUP"
    elif slug == "new-location":
        mutant["outputs"][0]["new_location"] = "FINAL"
    elif slug == "state":
        mutant["state"] = "SETUP"
    elif slug == "pending-kind":
        mutant["pending_kind"] = "PUBLISH"
    elif slug == "pending-index":
        mutant["pending_index"] = 1
    elif slug == "rollback-cursor":
        mutant["rollback_cursor"] = 1
    elif slug == "cleanup-cursor":
        mutant["cleanup_cursor"] = 1
    elif slug == "finalize-cursor":
        mutant["finalize_cursor"] = 1
    elif slug == "transition-id":
        mutant["transition_id"] = "T06"
    else:
        raise ValueError("unknown coherence metadata fixture " + slug)
    return coherence_serialize(low), coherence_serialize(mutant)


def mutate_transaction_journal(output: Path, mutation: str,
                               external: Path) -> None:
    transaction = output / TIMESTRETCH_TRANSACTION_DIRECTORY
    journals = [path for path in (transaction / "journal.a",
                                  transaction / "journal.b") if path.is_file()]
    if len(journals) != 2:
        raise RuntimeError("mutation fixture does not have two journals")

    def generation(path: Path) -> int:
        match = re.search(rb"(?m)^generation=([0-9]+)$", path.read_bytes())
        if match is None:
            raise RuntimeError("mutation fixture generation missing")
        return int(match.group(1))

    target = max(journals, key=generation)
    original = target.read_bytes()
    if mutation == "truncation":
        target.write_bytes(original[:max(1, len(original) // 2)])
    elif mutation == "oversize":
        target.write_bytes(b"X" * 65537)
    elif mutation == "duplicate-key":
        target.write_bytes(rewrite_transaction_checksum(
            original.replace(b"version=2\n", b"version=2\nversion=2\n", 1)))
    elif mutation == "unknown-key":
        target.write_bytes(rewrite_transaction_checksum(
            original.replace(b"state=", b"unknown_key=1\nstate=", 1)))
    elif mutation == "unknown-output-name":
        target.write_bytes(rewrite_transaction_checksum(
            original.replace(EXPECTED_OUTPUTS[0].encode("ascii"),
                             b"unknown-output.csv", 1)))
    elif mutation == "path-escape-name":
        target.write_bytes(rewrite_transaction_checksum(
            original.replace(EXPECTED_OUTPUTS[0].encode("ascii"),
                             b"../escape", 1)))
    elif mutation == "checksum-mismatch":
        changed = bytearray(original)
        changed[-2] = ord("0") if changed[-2] != ord("0") else ord("1")
        target.write_bytes(bytes(changed))
    elif mutation == "generation-gap":
        current = generation(target)
        replaced = re.sub(
            rb"(?m)^generation=[0-9]+$",
            "generation={}".format(current + 3).encode("ascii"), original,
            count=1)
        target.write_bytes(rewrite_transaction_checksum(replaced))
    elif mutation == "impossible-state-combination":
        replaced = re.sub(
            rb"(?m)^state=[A-Z_]+$",
            b"state=COMMITTED_CLEANUP_PENDING", original, count=1)
        target.write_bytes(rewrite_transaction_checksum(replaced))
    elif mutation == "symlink-substitution":
        external.write_bytes(original)
        target.unlink()
        target.symlink_to(external)
    else:
        raise ValueError("unknown journal mutation " + mutation)


def run_timestretch_durable_matrix(
    root: Path, scratch: Path, discovery: list[dict[str, object]],
) -> tuple[list[str], list[str]]:
    binaries, errors = compile_transaction_probes(root, scratch, discovery)
    observed: list[str] = []
    if errors:
        return errors, observed
    cases = scratch / "durable-cases"
    cases.mkdir()
    case_number = 0

    def fresh_case(seed: str, present: bool = True) \
            -> tuple[Path, dict[str, str], str]:
        nonlocal case_number
        case = cases / "case-{:04d}".format(case_number)
        case_number += 1
        case.mkdir()
        output = case / "output"
        old = write_transaction_fixture(output, seed, present)
        return output, old, "durable-new-{:04d}".format(case_number)

    def row_error(identity: str, detail: str) -> None:
        errors.append("TIMESTRETCH_DURABLE_ROW:{}:{}".format(identity, detail))

    def collision_oracle(identity: str, output: Path, binary: Path,
                         profile: str, detail: str) -> None:
        before = timestretch_tree_snapshot(output)
        result = run_transaction_probe(
            binary, output, "recover", profile, {}, output.parent)
        expected = (
            "ERROR TIMESTRETCH_TRANSACTION_RECOVERY_COLLISION "
            ".dspark-timestretch-transaction\n")
        if result.get("exit_code") != 6 \
                or result.get("stdout") != "" \
                or result.get("stderr") != expected \
                or not durable_process_clean(result) \
                or timestretch_tree_snapshot(output) != before:
            row_error(identity, detail)

    def pending_collision_control(identity: str, binary: Path,
                                  profile: str, kind: str) -> None:
        output, _old, seed = fresh_case(identity + "::" + kind)
        case = output.parent
        seeded = run_transaction_probe(
            binary, output, "run", profile,
            {"DSPARK_TIMESTRETCH_CRASH_CUT": "backup-before:00",
             "DSPARK_TIMESTRETCH_PROBE_SEED": seed}, case)
        if seeded.get("exit_code") != 86 or not durable_process_clean(seeded):
            row_error(identity, "PENDING_{}_SEED".format(kind.upper()))
            return
        transaction = output / TIMESTRETCH_TRANSACTION_DIRECTORY
        try:
            if kind == "both":
                shutil.copy2(output / EXPECTED_OUTPUTS[0],
                             transaction / "backup" / EXPECTED_OUTPUTS[0])
            elif kind == "neither":
                (output / EXPECTED_OUTPUTS[0]).unlink()
            elif kind == "wrong-hash":
                (output / EXPECTED_OUTPUTS[0]).unlink()
                (transaction / "backup" / EXPECTED_OUTPUTS[0]).write_bytes(
                    b"not the recorded old authority\n")
            elif kind == "unknown-internal":
                (transaction / "foreign.entry").write_bytes(b"foreign\n")
            elif kind == "transplanted-root":
                transplanted, _transplanted_old, _transplanted_seed = fresh_case(
                    identity + "::transplanted-destination")
                shutil.copytree(
                    transaction,
                    transplanted / TIMESTRETCH_TRANSACTION_DIRECTORY)
                output = transplanted
            else:
                raise ValueError("unknown pending collision control " + kind)
        except (OSError, ValueError) as error:
            row_error(identity, "PENDING_{}_FIXTURE:{}".format(
                kind.upper(), error))
            return
        collision_oracle(
            identity, output, binary, profile,
            "PENDING_{}_NOT_UNTOUCHED_COLLISION".format(kind.upper()))

    for compiler, profile in TIMESTRETCH_BUILD_LANES:
        binary = binaries[(compiler, profile)]
        lane = "{}::{}".format(compiler, profile)
        for mode in ("one-shot", "persistent"):
            for point in TIMESTRETCH_DURABLE_FAULT_POINTS:
                identity = "timestretch::{}::fault::{}::{}".format(
                    lane, mode, point)
                output, old, seed = fresh_case(identity)
                case = output.parent
                trace = case / "trace.log"
                base_values = {
                    "DSPARK_TIMESTRETCH_DURABLE_FAULT": point,
                    "DSPARK_TIMESTRETCH_DURABLE_MODE": mode,
                    "DSPARK_TIMESTRETCH_TRACE_FILE": str(trace),
                    "DSPARK_TIMESTRETCH_PROBE_SEED": seed,
                }
                operation = "run"
                retained = ""
                if durable_point_needs_rollback(point):
                    seeded = run_transaction_probe(
                        binary, output, "run", profile,
                        {"DSPARK_TIMESTRETCH_CRASH_CUT": "commit-ready",
                         "DSPARK_TIMESTRETCH_PROBE_SEED": seed}, case)
                    if seeded.get("exit_code") != 86 or not durable_process_clean(seeded):
                        row_error(identity, "ROLLBACK_SEED")
                    operation = "recover"
                    retained = "ROLLBACK"
                elif durable_point_needs_recovery_seed(point):
                    seeded = run_transaction_probe(
                        binary, output, "run", profile,
                        {"DSPARK_TIMESTRETCH_CRASH_CUT": "publish-after:00",
                         "DSPARK_TIMESTRETCH_PROBE_SEED": seed}, case)
                    if seeded.get("exit_code") != 86 or not durable_process_clean(seeded):
                        row_error(identity, "RECOVERY_SEED")
                    operation = "recover"
                    retained = "ROLLBACK"
                elif durable_point_is_postcommit(point):
                    retained = "CLEANUP"

                result = run_transaction_probe(
                    binary, output, operation, profile, base_values, case)
                trace_text = trace.read_text(encoding="ascii") \
                    if trace.is_file() else ""
                if "fault:{}:{}\n".format(point, mode) not in trace_text:
                    row_error(identity, "FAULT_NOT_REACHED")
                if not durable_process_clean(result):
                    row_error(identity, "PROCESS")

                if retained:
                    if not is_recovery_required(result, retained):
                        row_error(identity, "TYPED_RETAINED_TERMINAL")
                    if not timestretch_transaction_residue(output):
                        row_error(identity, "RECOVERY_STATE_MISSING")
                    if retained == "ROLLBACK" \
                            and not old_authority_retained(output, old):
                        row_error(identity, "OLD_AUTHORITY_LOST")
                    if retained == "CLEANUP" \
                            and timestretch_snapshot(output) \
                            != expected_probe_snapshot(seed):
                        row_error(identity, "NEW_AUTHORITY_LOST")
                    retained_snapshot = timestretch_tree_snapshot(output)
                    if mode == "persistent":
                        retry = run_transaction_probe(
                            binary, output, "recover", profile,
                            base_values, case)
                        if not is_recovery_required(retry, retained) \
                                or not durable_process_clean(retry) \
                                or timestretch_tree_snapshot(output) \
                                != retained_snapshot:
                            row_error(identity, "PERSISTENT_RETRY_NOT_IDEMPOTENT")
                    cleared = run_transaction_probe(
                        binary, output, "recover", profile,
                        {"DSPARK_TIMESTRETCH_PROBE_SEED": seed}, case)
                    expected_exit = 3 if retained == "ROLLBACK" else 0
                    expected_snapshot = old if retained == "ROLLBACK" \
                        else expected_probe_snapshot(seed)
                    if cleared.get("exit_code") != expected_exit \
                            or not durable_process_clean(cleared) \
                            or timestretch_snapshot(output) != expected_snapshot \
                            or timestretch_transaction_residue(output):
                        row_error(identity, "CLEARED_RETRY_DID_NOT_CONVERGE")
                else:
                    if result.get("exit_code") != 3 \
                            or not str(result.get("stderr", "")).startswith(
                                "ERROR TIMESTRETCH_TRANSACTION_") \
                            or timestretch_snapshot(output) != old \
                            or timestretch_transaction_residue(output):
                        row_error(identity, "PRECOMMIT_ABORT_ORACLE")
                observed.append(identity)

        for point in TIMESTRETCH_CRASH_POINTS:
            identity = "timestretch::{}::crash::{}".format(lane, point)
            output, old, seed = fresh_case(identity)
            case = output.parent
            trace = case / "trace.log"
            if crash_point_needs_rollback_seed(point):
                seeded = run_transaction_probe(
                    binary, output, "run", profile,
                    {"DSPARK_TIMESTRETCH_CRASH_CUT": "commit-ready",
                     "DSPARK_TIMESTRETCH_PROBE_SEED": seed}, case)
                if seeded.get("exit_code") != 86 or not durable_process_clean(seeded):
                    row_error(identity, "ROLLBACK_SEED")
                operation = "recover"
            else:
                operation = "run"
            crashed = run_transaction_probe(
                binary, output, operation, profile,
                {"DSPARK_TIMESTRETCH_CRASH_CUT": point,
                 "DSPARK_TIMESTRETCH_TRACE_FILE": str(trace),
                 "DSPARK_TIMESTRETCH_PROBE_SEED": seed}, case)
            trace_text = trace.read_text(encoding="ascii") \
                if trace.is_file() else ""
            if crashed.get("exit_code") != 86 \
                    or not durable_process_clean(crashed) \
                    or "crash:{}\n".format(point) not in trace_text:
                row_error(identity, "CRASH_CUT_NOT_REACHED")
            replay = run_transaction_probe(
                binary, output, "recover", profile,
                {"DSPARK_TIMESTRETCH_PROBE_SEED": seed}, case)
            committed = crash_point_is_postcommit(point)
            expected_exit = 0 if committed else 3
            expected_snapshot = expected_probe_snapshot(seed) if committed else old
            if replay.get("exit_code") != expected_exit \
                    or not durable_process_clean(replay) \
                    or timestretch_snapshot(output) != expected_snapshot \
                    or timestretch_transaction_residue(output):
                row_error(identity, "CRASH_REPLAY_AUTHORITY")
            stable = timestretch_tree_snapshot(output)
            second = run_transaction_probe(
                binary, output, "recover", profile, {}, case)
            if second.get("exit_code") != 0 or not durable_process_clean(second) \
                    or timestretch_tree_snapshot(output) != stable:
                row_error(identity, "CRASH_REPLAY_NOT_IDEMPOTENT")
            observed.append(identity)

        for mutation in TIMESTRETCH_JOURNAL_MUTATIONS:
            identity = "timestretch::{}::journal-mutation::{}".format(
                lane, mutation)
            output, _old, seed = fresh_case(identity)
            case = output.parent
            seeded = run_transaction_probe(
                binary, output, "run", profile,
                {"DSPARK_TIMESTRETCH_CRASH_CUT": "publish-after:00",
                 "DSPARK_TIMESTRETCH_PROBE_SEED": seed}, case)
            if seeded.get("exit_code") != 86 or not durable_process_clean(seeded):
                row_error(identity, "MUTATION_SEED")
            try:
                mutate_transaction_journal(
                    output, mutation, case / "external-journal")
            except (OSError, ValueError, RuntimeError) as error:
                row_error(identity, "MUTATION_FIXTURE:" + str(error))
            before = timestretch_tree_snapshot(output)
            result = run_transaction_probe(
                binary, output, "recover", profile, {}, case)
            if result.get("exit_code") != 6 \
                    or "ERROR TIMESTRETCH_TRANSACTION_RECOVERY_COLLISION " \
                    not in str(result.get("stderr")) \
                    or not durable_process_clean(result) \
                    or timestretch_tree_snapshot(output) != before:
                row_error(identity, "MUTATION_NOT_UNTOUCHED_COLLISION")
            pending_kind = {
                "truncation": "both",
                "oversize": "neither",
                "duplicate-key": "wrong-hash",
                "unknown-key": "unknown-internal",
                "unknown-output-name": "transplanted-root",
            }.get(mutation)
            if pending_kind is not None:
                pending_collision_control(
                    identity, binary, profile, pending_kind)
            observed.append(identity)

        for baseline in ("fresh", "replacement"):
            identity = "timestretch::{}::baseline::{}".format(lane, baseline)
            output, _old, seed = fresh_case(identity, baseline == "replacement")
            case = output.parent
            result = run_transaction_probe(
                binary, output, "run", profile,
                {"DSPARK_TIMESTRETCH_PROBE_SEED": seed}, case)
            if result.get("exit_code") != 0 \
                    or result.get("stdout") != "" or result.get("stderr") != "" \
                    or not durable_process_clean(result) \
                    or timestretch_snapshot(output) != expected_probe_snapshot(seed) \
                    or timestretch_transaction_residue(output):
                row_error(identity, "BASELINE")
            if baseline == "replacement":
                mixed, _mixed_old, mixed_seed = fresh_case(
                    identity + "::mixed-old-and-absent", False)
                for index, name in enumerate(EXPECTED_OUTPUTS):
                    if index % 2 == 0:
                        (mixed / name).write_bytes(
                            "DSPark mixed old {} {} {}\n".format(
                                mixed_seed, index, name).encode("ascii"))
                mixed_before = timestretch_tree_snapshot(mixed)
                mixed_result = run_transaction_probe(
                    binary, mixed, "run", profile,
                    {"DSPARK_TIMESTRETCH_DURABLE_FAULT":
                         "publish-rename:04",
                     "DSPARK_TIMESTRETCH_DURABLE_MODE": "one-shot",
                     "DSPARK_TIMESTRETCH_PROBE_SEED": mixed_seed},
                    mixed.parent)
                if mixed_result.get("exit_code") != 3 \
                        or not durable_process_clean(mixed_result) \
                        or timestretch_tree_snapshot(mixed) != mixed_before \
                        or timestretch_transaction_residue(mixed):
                    row_error(identity, "MIXED_OLD_ABSENCE_ROLLBACK")
            observed.append(identity)
    return errors, observed


def run_timestretch_coherence_matrix(
    root: Path, scratch: Path, discovery: list[dict[str, object]],
) -> tuple[list[str], list[str]]:
    binaries, errors = compile_coherence_probes(root, scratch, discovery)
    observed: list[str] = []
    if errors:
        return errors, observed
    cases = scratch / "coherence-cases"
    cases.mkdir()
    case_number = 0

    def fresh_case() -> Path:
        nonlocal case_number
        case = cases / "case-{:04d}".format(case_number)
        case_number += 1
        case.mkdir()
        return case

    def row_error(identity: str, detail: str) -> None:
        errors.append("TIMESTRETCH_COHERENCE_ROW:{}:{}".format(
            identity, detail))

    def run_probe(binary: Path, profile: str,
                  arguments: list[str], cwd: Path) -> dict[str, object]:
        return run_child(
            [str(binary), *arguments], cwd, 30,
            transaction_environment(profile))

    for compiler, profile in TIMESTRETCH_BUILD_LANES:
        binary = binaries[(compiler, profile)]
        lane = "{}::{}".format(compiler, profile)
        for slug in TIMESTRETCH_COHERENCE_PAIR_CASES:
            identity = "timestretch::{}::coherence-pair::{}".format(
                lane, slug)
            case = fresh_case()
            root_binding, transaction_id = coherence_root_identity(case)
            finals, temporaries, expected = coherence_pair_fixture(
                slug, root_binding, transaction_id)
            for path, value in zip(
                    (case / "journal.a", case / "journal.b"), finals):
                if value is not None:
                    path.write_bytes(value)
            for path, value in zip(
                    (case / "journal.a.tmp", case / "journal.b.tmp"),
                    temporaries):
                if value is not None:
                    path.write_bytes(value)
            before = timestretch_tree_snapshot(case)
            result = run_probe(binary, profile, ["pair", str(case)], case)
            accepted = expected == "ACCEPT"
            expected_exit = 0 if accepted else 6
            expected_stdout = "ACCEPT\n" if accepted else ""
            expected_stderr = "" if accepted else (
                "ERROR TIMESTRETCH_TRANSACTION_RECOVERY_COLLISION "
                ".dspark-timestretch-transaction\n")
            if result.get("exit_code") != expected_exit \
                    or result.get("stdout") != expected_stdout \
                    or result.get("stderr") != expected_stderr \
                    or not durable_process_clean(result) \
                    or timestretch_tree_snapshot(case) != before:
                row_error(identity, "PAIR_CLASSIFICATION_OR_TREE")
            observed.append(identity)

        for transition_id, _source, _target, _event \
                in TIMESTRETCH_COHERENCE_TRANSITIONS:
            identity = "timestretch::{}::coherence-transition::{}".format(
                lane, transition_id)
            case = fresh_case()
            root_binding, transaction_id = coherence_root_identity(case)
            low_record = coherence_fixture(
                transition_id, root_binding, transaction_id)
            high_record = coherence_apply(low_record, transition_id)
            low = case / "low.journal"
            high = case / "high.journal"
            low.write_bytes(coherence_serialize(low_record))
            high.write_bytes(coherence_serialize(high_record))
            result = run_probe(
                binary, profile, ["transition", str(low), str(high)], case)
            if result.get("exit_code") != 0 \
                    or result.get("stdout") != "ACCEPT\n" \
                    or result.get("stderr") != "" \
                    or not durable_process_clean(result):
                row_error(identity, "LEGAL_TRANSITION_REJECTED")
            observed.append(identity)

        for slug in TIMESTRETCH_COHERENCE_METADATA_CASES:
            identity = "timestretch::{}::coherence-metadata::{}".format(
                lane, slug)
            case = fresh_case()
            root_binding, transaction_id = coherence_root_identity(case)
            low_bytes, high_bytes = coherence_metadata_fixture(
                slug, root_binding, transaction_id)
            low = case / "low.journal"
            high = case / "high.journal"
            low.write_bytes(low_bytes)
            high.write_bytes(high_bytes)
            before = timestretch_tree_snapshot(case)
            result = run_probe(
                binary, profile, ["transition", str(low), str(high)], case)
            if result.get("exit_code") != 0 \
                    or result.get("stdout") != "REJECT\n" \
                    or result.get("stderr") != "" \
                    or not durable_process_clean(result) \
                    or timestretch_tree_snapshot(case) != before:
                row_error(identity, "METADATA_DIVERGENCE_OR_TREE")
            observed.append(identity)
    return errors, observed


def replace_source_assignment(source: str, name: str,
                              replacement: str) -> str:
    tree = ast.parse(source)
    nodes: list[ast.AST] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == name
               for target in targets):
            nodes.append(node)
    if len(nodes) != 1 or nodes[0].end_lineno is None:
        raise ValueError("assignment not unique: " + name)
    lines = source.splitlines(keepends=True)
    lines[nodes[0].lineno - 1:nodes[0].end_lineno] = [replacement + "\n"]
    return "".join(lines)


def replace_source_function(source: str, name: str,
                            replacement: str) -> str:
    tree = ast.parse(source)
    nodes = [node for node in ast.walk(tree)
             if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(nodes) != 1 or nodes[0].end_lineno is None:
        raise ValueError("function not unique: " + name)
    lines = source.splitlines(keepends=True)
    lines[nodes[0].lineno - 1:nodes[0].end_lineno] = [replacement + "\n"]
    return "".join(lines)


def replace_source_function_fragment(
    source: str, name: str, target: str, replacement: str,
) -> str | None:
    tree = ast.parse(source)
    nodes = [node for node in ast.walk(tree)
             if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(nodes) != 1 or nodes[0].end_lineno is None:
        return None
    lines = source.splitlines(keepends=True)
    begin = nodes[0].lineno - 1
    end = nodes[0].end_lineno
    fragment = "".join(lines[begin:end])
    if fragment.count(target) != 1:
        return None
    mutant = fragment.replace(target, replacement, 1)
    if mutant == fragment or target in mutant:
        return None
    lines[begin:end] = [mutant]
    return "".join(lines)


def timestretch_outer_mutant_controls(
    root: Path, source: str,
) -> list[tuple[str, bool]]:
    controls: list[tuple[str, bool]] = []

    def structure_control(slug: str, mutant: str, terminal: str) -> None:
        controls.append((
            slug, terminal in timestretch_durable_structure_errors(mutant)))

    structure_control(
        "delete-durable-recovery-invocation",
        source.replace(
            "        durable_errors, durable_observed = "
            "run_timestretch_durable_matrix(\n"
            "            root, scratch, discovery)\n",
            "        durable_errors, durable_observed = [], []\n", 1),
        "TIMESTRETCH_DURABLE_RECOVERY_INVOCATION")
    structure_control(
        "delete-literal-fault-inventory",
        replace_source_assignment(
            source, "TIMESTRETCH_DURABLE_FAULT_POINTS",
            "TIMESTRETCH_DURABLE_FAULT_POINTS_REMOVED = ()"),
        "TIMESTRETCH_DURABLE_LITERAL_FAULT_INVENTORY")
    structure_control(
        "replace-literal-inventory-with-textual-marker",
        replace_source_assignment(
            source, "TIMESTRETCH_DURABLE_FAULT_POINTS",
            'TIMESTRETCH_DURABLE_FAULT_POINTS = "193 fault positions"'),
        "TIMESTRETCH_DURABLE_LITERAL_FAULT_INVENTORY")
    structure_control(
        "delete-observed-versus-expected-identity-comparison",
        source.replace(
            "        if observed != list(EXPECTED_TIMESTRETCH_EXECUTION_IDS):\n",
            "        if False:\n", 1),
        "TIMESTRETCH_DURABLE_OBSERVED_IDENTITY_COMPARISON")
    structure_control(
        "delete-journal-mutation-loop",
        source.replace(
            "        for mutation in TIMESTRETCH_JOURNAL_MUTATIONS:\n",
            "        for mutation in ():\n", 1),
        "TIMESTRETCH_DURABLE_JOURNAL_MUTATION_LOOP")
    crash_mutant = replace_source_function_fragment(
        source, "run_timestretch_durable_matrix",
        "        for point in TIMESTRETCH_CRASH_POINTS:\n",
        "        for point in ():\n")
    if crash_mutant is None:
        controls.append(("delete-crash-cut-loop", False))
    else:
        structure_control(
            "delete-crash-cut-loop", crash_mutant,
            "TIMESTRETCH_DURABLE_CRASH_LOOP")
    structure_control(
        "delete-ordinary-success-suppression-oracle",
        replace_source_function(
            source, "durable_process_clean",
            "def durable_process_clean(result: dict[str, object]) -> bool:\n"
            "    combined = str(result.get('stdout')) + str(result.get('stderr'))\n"
            "    return result.get('errors') == [] and not any(\n"
            "        item in combined for item in SANITIZER_DIAGNOSTICS)"),
        "TIMESTRETCH_DURABLE_SUCCESS_SUPPRESSION_ORACLE")
    structure_control(
        "delete-recovery-terminal-oracle",
        source.replace(
            "                    if not is_recovery_required(result, retained):\n",
            "                    if False:\n", 1),
        "TIMESTRETCH_DURABLE_RECOVERY_TERMINAL_ORACLE")
    structure_control(
        "delete-inherited-legacy-loop",
        source.replace(
            "            for phase, variable in TIMESTRETCH_FAILURE_PHASES:\n",
            "            for phase, variable in ():\n", 1),
        "TIMESTRETCH_DURABLE_LEGACY_LOOP")

    cmake_path = "tests/CMakeLists.txt"
    cmake = (root / cmake_path).read_text(encoding="ascii")
    changed_cmake = cmake.replace(
        "${PROJECT_SOURCE_DIR}/tools/verify_global_validation_contract.py",
        "${PROJECT_SOURCE_DIR}/tools/verify_m018_global_corrections.py", 1)
    controls.append(("delete-ctest-binding",
                     bool(binding_errors(root, {cmake_path: changed_cmake}))))
    for slug, path in (("delete-ci-binding", ".github/workflows/ci.yml"),
                       ("delete-docs-binding", ".github/workflows/docs.yml")):
        original = (root / path).read_text(encoding="ascii")
        controls.append((
            slug, bool(binding_errors(
                root, {path: original.replace(LIVE_COMMAND, "", 1)}))))
    return controls


def timestretch_coherence_outer_mutant_controls(
    root: Path, source: str, characterizer_source: str,
) -> list[tuple[str, bool]]:
    controls: list[tuple[str, bool]] = []

    def structure_control(
        slug: str,
        mutant_source: str,
        mutant_characterizer: str,
        terminal: str,
    ) -> None:
        controls.append((
            slug,
            terminal in timestretch_coherence_structure_errors(
                mutant_source, mutant_characterizer),
        ))

    def exact_once(text: str, target: str, replacement: str) -> str | None:
        if text.count(target) != 1:
            return None
        mutant = text.replace(target, replacement, 1)
        return mutant if mutant != text and mutant.count(target) == 0 else None

    structure_control(
        "delete-pair-validator", source,
        characterizer_source.replace(
            "bool transactionClassifyAuthority(",
            "bool transactionClassifyAuthorityRemoved(", 1),
        "TIMESTRETCH_COHERENCE_PAIR_VALIDATOR")
    structure_control(
        "delete-predecessor-binding", source,
        characterizer_source.replace(
            "high.journal.previousPayloadSha256 != low.payloadSha256",
            "false", 1),
        "TIMESTRETCH_COHERENCE_PREDECESSOR_BINDING")
    structure_control(
        "delete-metadata-comparison", source,
        characterizer_source.replace(
            "transactionJournalBodyEqual(high, supplied)", "true", 1),
        "TIMESTRETCH_COHERENCE_METADATA_COMPARISON")
    transition_target = (
        "if transition_value != TIMESTRETCH_COHERENCE_TRANSITIONS \\\n"
        "            or len(transition_value or ()) != 32:")
    transition_mutant = exact_once(source, transition_target, "if False:")
    if transition_mutant is None:
        controls.append(("delete-transition-inventory-comparison", False))
    else:
        structure_control(
            "delete-transition-inventory-comparison", transition_mutant,
            characterizer_source,
            "TIMESTRETCH_COHERENCE_TRANSITION_INVENTORY_COMPARISON")
    execution_target = (
        "        if observed_rows != "
        "list(EXPECTED_TIMESTRETCH_EXECUTION_ROWS):\n")
    execution_mutant = exact_once(
        source, execution_target, "        if False:\n")
    if execution_mutant is None:
        controls.append(("delete-execution-inventory-comparison", False))
    else:
        structure_control(
            "delete-execution-inventory-comparison", execution_mutant,
            characterizer_source,
            "TIMESTRETCH_COHERENCE_EXECUTION_INVENTORY_COMPARISON")

    cmake_path = "tests/CMakeLists.txt"
    cmake = (root / cmake_path).read_text(encoding="ascii")
    changed_cmake = cmake.replace(
        "${PROJECT_SOURCE_DIR}/tools/verify_global_validation_contract.py",
        "${PROJECT_SOURCE_DIR}/tools/verify_m018_global_corrections.py", 1)
    controls.append((
        "delete-ctest-binding",
        bool(binding_errors(root, {cmake_path: changed_cmake}))))
    for slug, path in (
        ("delete-ci-workflow-binding", ".github/workflows/ci.yml"),
        ("delete-docs-workflow-binding", ".github/workflows/docs.yml"),
    ):
        original = (root / path).read_text(encoding="ascii")
        controls.append((
            slug,
            bool(binding_errors(
                root, {path: original.replace(LIVE_COMMAND, "", 1)}))))

    for transition in TIMESTRETCH_COHERENCE_TRANSITIONS:
        transition_id = transition[0]
        mutant_rows = tuple(
            row for row in TIMESTRETCH_COHERENCE_TRANSITIONS
            if row[0] != transition_id)
        mutant_source = replace_source_assignment(
            source, "TIMESTRETCH_COHERENCE_TRANSITIONS",
            "TIMESTRETCH_COHERENCE_TRANSITIONS = " + repr(mutant_rows))
        structure_control(
            "delete-transition-row::" + transition_id,
            mutant_source, characterizer_source,
            "TIMESTRETCH_COHERENCE_TRANSITION_INVENTORY")
    return controls


def timestretch_live(root: Path,
                     discovery: list[dict[str, object]]) -> list[str]:
    errors: list[str] = []
    legacy_observed: list[str] = []
    with tempfile.TemporaryDirectory(
            prefix="dspark-timestretch-live-") as directory:
        scratch = Path(directory)
        for compiler in discovery:
            label = str(compiler["label"])
            binary = scratch / ("characterize-" + label)
            instrumented = scratch / ("characterize-transaction-" + label)
            base_command = [
                str(compiler["path"]), "-std=c++20", "-O2", "-Wall",
                "-Wextra", "-Wpedantic", "-Werror", "-I", str(root),
                str(root / "tools/characterize_timestretch.cpp"),
            ]
            build = producer.run_owned(
                [*base_command, "-o", str(binary)], scratch, 240, "compile")
            test_build = producer.run_owned(
                [*base_command,
                 "-DDSPARK_TIMESTRETCH_TRANSACTION_TESTING=1",
                 "-o", str(instrumented)], scratch, 240, "compile")
            if build["exit_code"] != 0 or build["errors"]:
                errors.append("TIMESTRETCH_BUILD:" + label)
                continue
            if test_build["exit_code"] != 0 or test_build["errors"]:
                errors.append("TIMESTRETCH_INSTRUMENTED_BUILD:" + label)
                continue

            missing = scratch / ("missing-" + label)
            result = producer.run_owned(
                [str(binary), str(missing)], scratch, 180, "run")
            if result["exit_code"] != 2 \
                    or "ERROR TIMESTRETCH_OUTPUT_DIRECTORY" not in result["stderr"] \
                    or "characterisation written" in result["stdout"] \
                    or missing.exists():
                errors.append("TIMESTRETCH_MISSING_DIRECTORY:" + label)
            regular = scratch / ("regular-" + label)
            regular.write_bytes(b"not a directory\n")
            result = producer.run_owned(
                [str(binary), str(regular)], scratch, 180, "run")
            if result["exit_code"] != 2 \
                    or "ERROR TIMESTRETCH_OUTPUT_DIRECTORY" not in result["stderr"] \
                    or "characterisation written" in result["stdout"]:
                errors.append("TIMESTRETCH_REGULAR_FILE:" + label)

            nonregular = scratch / ("nonregular-final-" + label)
            nonregular.mkdir()
            (nonregular / EXPECTED_OUTPUTS[0]).mkdir()
            result = producer.run_owned(
                [str(binary), str(nonregular)], scratch, 180, "run")
            if result["exit_code"] != 3 \
                    or result["stderr"] != (
                        "ERROR TIMESTRETCH_TRANSACTION_PRECHECK {}\n".format(
                            EXPECTED_OUTPUTS[0])) \
                    or not (nonregular / EXPECTED_OUTPUTS[0]).is_dir() \
                    or len(list(nonregular.iterdir())) != 1:
                errors.append("TIMESTRETCH_NONREGULAR_FINAL:" + label)

            symlink_root = scratch / ("symlink-final-" + label)
            symlink_root.mkdir()
            symlink_target = scratch / ("symlink-target-" + label)
            symlink_target.write_bytes(b"caller-owned symlink target\n")
            (symlink_root / EXPECTED_OUTPUTS[0]).symlink_to(symlink_target)
            target_before = hashlib.sha256(symlink_target.read_bytes()).hexdigest()
            result = producer.run_owned(
                [str(binary), str(symlink_root)], scratch, 180, "run")
            if result["exit_code"] != 3 \
                    or result["stderr"] != (
                        "ERROR TIMESTRETCH_TRANSACTION_PRECHECK {}\n".format(
                            EXPECTED_OUTPUTS[0])) \
                    or not (symlink_root / EXPECTED_OUTPUTS[0]).is_symlink() \
                    or hashlib.sha256(symlink_target.read_bytes()).hexdigest() \
                    != target_before:
                errors.append("TIMESTRETCH_SYMLINK_FINAL:" + label)

            collision = scratch / ("collision-" + label)
            collision_before = write_timestretch_sentinels(
                collision, label + "-collision")
            (collision / TIMESTRETCH_TRANSACTION_DIRECTORY).mkdir()
            result = producer.run_owned(
                [str(binary), str(collision)], scratch, 180, "run")
            (collision / TIMESTRETCH_TRANSACTION_DIRECTORY).rmdir()
            if result["exit_code"] != 6 \
                    or result["stderr"] != (
                        "ERROR TIMESTRETCH_TRANSACTION_RECOVERY_COLLISION {}\n".format(
                            TIMESTRETCH_TRANSACTION_DIRECTORY)) \
                    or timestretch_snapshot(collision) != collision_before:
                errors.append("TIMESTRETCH_STAGE_COLLISION:" + label)

            for phase, variable in TIMESTRETCH_FAILURE_PHASES:
                for index, output_name in enumerate(EXPECTED_OUTPUTS):
                    failure_root = scratch / (
                        "{}-failure-{}-{}".format(
                            phase.lower(), label, index))
                    before = write_timestretch_sentinels(
                        failure_root,
                        "{}-{}-{}".format(label, phase.lower(), index))
                    environment = dict(os.environ)
                    for _other_phase, other_variable \
                            in TIMESTRETCH_FAILURE_PHASES:
                        environment.pop(other_variable, None)
                    environment[variable] = str(index)
                    result = producer.run_owned(
                        [str(instrumented), str(failure_root)],
                        scratch, 180, "run", environment)
                    errors.extend(validate_timestretch_rollback(
                        failure_root, before, result, phase, output_name))
                    legacy_observed.append(
                        "timestretch::{}::normal::legacy-{}::{:02d}".format(
                            label, phase.lower(), index))

            output = scratch / ("output-" + label)
            output.mkdir()
            production_environment = dict(os.environ)
            production_environment.update({
                "DSPARK_TIMESTRETCH_FAIL_STAGE_INDEX": "0",
                "DSPARK_TIMESTRETCH_FAIL_COMMIT_INDEX": "0",
                "DSPARK_TIMESTRETCH_DURABLE_FAULT": "journal-write:setup",
                "DSPARK_TIMESTRETCH_DURABLE_MODE": "persistent",
                "DSPARK_TIMESTRETCH_CRASH_CUT": "setup-journal",
            })
            result = producer.run_owned(
                [str(binary), str(output)], scratch, 180, "run",
                production_environment)
            names = sorted(path.name for path in output.iterdir())
            if result["exit_code"] != 0 or result["errors"] \
                    or result["stderr"] != "" \
                    or result["stdout"] != (
                        "characterisation written to {}\n".format(output)) \
                    or names != sorted(EXPECTED_OUTPUTS) \
                    or timestretch_transaction_residue(output):
                errors.append("TIMESTRETCH_SUCCESS_TRANSACTION:" + label)
                continue
            first_success = timestretch_snapshot(output)
            if first_success is None:
                errors.append("TIMESTRETCH_OUTPUT_ARTIFACT:" + label)
                continue
            rerun = producer.run_owned(
                [str(binary), str(output)], scratch, 180, "run")
            if rerun["exit_code"] != 0 or rerun["errors"] \
                    or rerun["stderr"] != "" \
                    or timestretch_snapshot(output) != first_success \
                    or timestretch_transaction_residue(output):
                errors.append("TIMESTRETCH_SUCCESSFUL_REPLACEMENT:" + label)
            errors.extend(
                error + ":" + label for error in measurement_errors(output))
        durable_errors, durable_observed = run_timestretch_durable_matrix(
            root, scratch, discovery)
        errors.extend(durable_errors)
        coherence_errors, coherence_observed = \
            run_timestretch_coherence_matrix(root, scratch, discovery)
        errors.extend(coherence_errors)
        source = Path(__file__).read_text(encoding="ascii")
        characterizer_source = (
            root / "tools/characterize_timestretch.cpp"
        ).read_text(encoding="ascii")
        outer_controls = timestretch_outer_mutant_controls(root, source)
        outer_observed: list[str] = []
        for slug, passed in outer_controls:
            if not passed:
                errors.append("TIMESTRETCH_OUTER_MUTANT:" + slug)
            outer_observed.append("timestretch::outer-mutant::" + slug)
        coherence_controls = timestretch_coherence_outer_mutant_controls(
            root, source, characterizer_source)
        coherence_outer_observed: list[str] = []
        for slug, passed in coherence_controls:
            if not passed:
                errors.append("TIMESTRETCH_COHERENCE_OUTER_MUTANT:" + slug)
            coherence_outer_observed.append(
                "timestretch::coherence-outer-mutant::" + slug)
        observed = (durable_observed + legacy_observed + outer_observed
                    + coherence_observed + coherence_outer_observed)
        observed_rows = [
            EXPECTED_TIMESTRETCH_EXECUTION_ROW_BY_ID.get(identity)
            for identity in observed
        ]
        if observed != list(EXPECTED_TIMESTRETCH_EXECUTION_IDS):
            errors.append("TIMESTRETCH_EXECUTION_IDENTITY_ORDER_CARDINALITY")
        if observed_rows != list(EXPECTED_TIMESTRETCH_EXECUTION_ROWS):
            errors.append("TIMESTRETCH_EXECUTION_ROW_ORDER_CONTENT")
    return errors


def flac_live(root: Path,
              discovery: list[dict[str, object]]) -> list[str]:
    errors: list[str] = []
    with tempfile.TemporaryDirectory(prefix="dspark-flac-live-") as directory:
        scratch = Path(directory)
        for compiler in discovery:
            label = str(compiler["label"])
            command = [
                str(compiler["path"]), "-std=c++20", "-O3", "-Wall",
                "-Wextra", "-Wpedantic", "-Werror",
                "-fno-omit-frame-pointer",
                "-fsanitize=address,undefined,float-cast-overflow",
                "-fno-sanitize-recover=all",
                '-DDSPARK_TEST_FIXTURE_DIR="{}"'.format(
                    root / "tests/fixtures"),
                "-I", str(root), "-c", str(root / "tests/TestIO.cpp"),
                "-o", str(scratch / ("TestIO-" + label + ".o")),
            ]
            result = producer.run_owned(command, scratch, 240, "compile")
            if result["exit_code"] != 0 or result["errors"]:
                errors.append("FLAC_STRICT_COMPILE:" + label)
    return errors


def live_mode(root: Path) -> int:
    if os.name != "posix" or not Path("/proc").is_dir():
        print("ERROR UNSUPPORTED_PROCESS_TREE_PLATFORM", file=sys.stderr)
        return 1
    if ctest_mode(root) != 0:
        return 1
    with tempfile.TemporaryDirectory(
            prefix="dspark-global-outer-") as directory:
        transcript_path = Path(directory) / "producer-transcript.json"
        producer_result = run_live_producer(root, transcript_path)
        if not transcript_path.is_file():
            print("ERROR VALIDATION_OUTER_TRANSCRIPT_MISSING", file=sys.stderr)
            return 1
        try:
            transcript = json.loads(transcript_path.read_text(encoding="ascii"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            print("ERROR VALIDATION_OUTER_TRANSCRIPT_PARSE " + str(error),
                  file=sys.stderr)
            return 1
        producer_exit_value = producer_result.get("exit_code")
        producer_exit = producer_exit_value \
            if isinstance(producer_exit_value, int) else -1
        errors = validate_transcript(transcript, producer_exit)
        if producer_result.get("errors"):
            errors.append("VALIDATION_OUTER_PRODUCER_PROCESS:" + str(
                producer_result["errors"]))
        discovery = transcript.get("compiler_discovery", [])
        if not errors:
            errors.extend(generic_gcc_live(root))
        if not errors:
            errors.extend(threading_external_live(root))
        if not errors:
            errors.extend(timestretch_live(root, discovery))
            errors.extend(flac_live(root, discovery))
        for error in errors:
            print("ERROR " + error, file=sys.stderr)
        if errors:
            return 1
    print("PASS global validation contract live: 2 primary discovery, "
          "1 generic GCC13 fallback, 4 Reverb sources, 16 Reverb rows, "
          "33 threading mutations, 2484 TimeStretch rows "
          "(1544 faults, 460 crashes, 40 mutations, 8 baselines, "
          "36 legacy, 12 inherited outer mutants, 144 coherence pairs, "
          "128 coherence transitions, 72 coherence metadata mutations, "
          "40 coherence outer mutants), 4 green Reverb baselines, "
          "12 exact Reverb expected REDs")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--ctest", action="store_true")
    modes.add_argument("--live", action="store_true")
    modes.add_argument("--self-test", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    package_r_outer_bootstrap = globals().get(
        "package_r_outer_executable_bootstrap_errors")
    if not callable(package_r_outer_bootstrap):
        package_r_outer_bootstrap_errors = [
            "PACKAGE_R_ORACLE_OUTER_BOOTSTRAP_HELPER_MISSING"]
    else:
        try:
            package_r_outer_source = Path(__file__).read_text(encoding="ascii")
            package_r_outer_producer_source = (
                Path(__file__).resolve().parent
                / "verify_m018_global_corrections.py"
            ).read_text(encoding="ascii")
            package_r_outer_bootstrap_errors = package_r_outer_bootstrap(
                package_r_outer_source, package_r_outer_producer_source)
        except (AttributeError, NameError, OSError, SyntaxError, TypeError,
                UnicodeError, ValueError) as error:
            package_r_outer_bootstrap_errors = [
                "PACKAGE_R_ORACLE_OUTER_BOOTSTRAP_INVOCATION:"
                + type(error).__name__]
    if not isinstance(package_r_outer_bootstrap_errors, list) \
            or not all(isinstance(item, str)
                       for item in package_r_outer_bootstrap_errors):
        package_r_outer_bootstrap_errors = [
            "PACKAGE_R_ORACLE_OUTER_BOOTSTRAP_RESULT"]
    if package_r_outer_bootstrap_errors:
        for error in package_r_outer_bootstrap_errors:
            print("ERROR " + error, file=sys.stderr)
        return 1

    if arguments.self_test:
        return self_test(root)
    if arguments.ctest:
        return ctest_mode(root)
    return live_mode(root)


if __name__ == "__main__":
    raise SystemExit(main())
