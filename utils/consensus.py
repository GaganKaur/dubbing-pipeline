"""utils/consensus.py — Merge N QA agent reports into one consensus report."""
from collections import defaultdict

from config.settings import settings
from utils.segment_models import QAReport, QAIssue, ConsensusQAReport, IssueSeverity


SEVERITY_WEIGHT = {
    IssueSeverity.SEVERE: 3,
    IssueSeverity.MODERATE: 2,
    IssueSeverity.MINOR: 1,
}


def merge_qa_reports(
    reports: list[QAReport],
    agreement_threshold: float = 0.5,
) -> ConsensusQAReport:
    """
    Merge N independent QA agent reports for the same segment.

    Rules:
    - Issues flagged by >= threshold fraction of agents → included at full severity
    - Issues flagged by < threshold fraction → downgraded to MINOR (FYI only)
    - Consensus score = weighted average of agent scores
    - passed = True only if consensus_score >= qa_threshold AND no SEVERE/MODERATE issues
    """
    if not reports:
        raise ValueError("Cannot merge empty report list")

    n = len(reports)
    segment_id = reports[0].segment_id
    iteration = reports[0].iteration

    # ── Average score ────────────────────────────────────────────────────
    consensus_score = sum(r.score for r in reports) / n

    # ── Group issues by type ──────────────────────────────────────────────
    # Independent agents rarely word a description identically, but agreement
    # on *what kind* of problem a segment has (e.g. tone_mismatch) is the
    # meaningful signal — group on issue_type rather than exact text.
    issue_groups: dict[str, list[QAIssue]] = defaultdict(list)
    for report in reports:
        for issue in report.issues:
            issue_groups[issue.issue_type].append(issue)

    # ── Apply agreement threshold ─────────────────────────────────────────
    merged_issues: list[QAIssue] = []
    for issue_type, issue_list in issue_groups.items():
        agreement = len(issue_list) / n
        # Use the most detailed description as the representative
        representative = max(issue_list, key=lambda i: len(i.description))

        if agreement >= agreement_threshold:
            # Majority agreement — keep at reported severity
            # Escalate if any agent flagged higher
            max_severity = max(
                issue_list, key=lambda i: SEVERITY_WEIGHT[i.severity]
            ).severity
            merged_issues.append(
                QAIssue(
                    issue_id=representative.issue_id,
                    segment_id=segment_id,
                    issue_type=representative.issue_type,
                    severity=max_severity,
                    description=representative.description,
                    fix_hint=_best_fix_hint(issue_list),
                )
            )
        else:
            # Minority flag — downgrade to MINOR as FYI
            merged_issues.append(
                QAIssue(
                    issue_id=representative.issue_id,
                    segment_id=segment_id,
                    issue_type=representative.issue_type,
                    severity=IssueSeverity.MINOR,
                    description=f"[FYI — minority flag] {representative.description}",
                    fix_hint=representative.fix_hint,
                )
            )

    # ── Passed check ─────────────────────────────────────────────────────
    # Passed only if no SEVERE/MODERATE issues remain after consensus AND
    # the averaged score clears the configured threshold.
    blocking_issues = [
        i for i in merged_issues
        if i.severity in (IssueSeverity.SEVERE, IssueSeverity.MODERATE)
    ]
    passed = len(blocking_issues) == 0 and consensus_score >= settings.qa_threshold

    return ConsensusQAReport(
        iteration=iteration,
        segment_id=segment_id,
        consensus_score=round(consensus_score, 3),
        issues=merged_issues,
        passed=passed,
    )


def _best_fix_hint(issues: list[QAIssue]) -> str:
    """Pick the most specific fix hint from a group of issues about the same problem."""
    # Prefer the longest / most detailed hint
    return max(issues, key=lambda i: len(i.fix_hint)).fix_hint
