---
layout: post
title: "Two Small Claude Code Skills for Code Review and Research"
date: 2026-11-12 12:00:00 -0500
---

I recently built two Claude Code skills to handle tasks I kept repeating: getting a second code review and checking the literature before making a research plan.

### Cursor for a second code review

I already use Cursor to review code written by other agents. What I wanted was a simple skill with clearer limits on what the reviewer could access.

The [Cursor Review Skill](https://github.com/hdeng26/claude-cursor-review-skill) lets me type:

```text
/cursor
```

Claude prepares the changes and a short explanation of the task, then asks Cursor to review them. That context matters: a reviewer needs to know what the code was supposed to do.

The skill runs Cursor in Ask mode inside a temporary workspace, with Linux bubblewrap restricting access to the surrounding filesystem. It returns the review without automatically fixing or merging anything. It runs only when I request it.

This limits local access, but the supplied code and context still go to Cursor’s service. A second review can also miss bugs; it is another useful check.

### Gemini for research background

The other motivation was practical: my Google AI Pro subscription includes [$10 in monthly developer credits](https://developers.google.com/program/plans-and-pricing), which encouraged me to make more use of API-based research tools.

The [Gemini Research Skill](https://github.com/hdeng26/claude-gemini-research-skill) connects Claude Code to Gemini and Google Search. My main use is:

```text
/gemini-research background "self-supervised representation learning"
```

Before drafting a research plan, it searches related work, selects a few relevant papers, reads available full texts, and produces a brief with supporting evidence.

I did not want to rely on a plausible-looking list of papers. The scripts therefore check paper metadata against research databases and check quoted passages against the retrieved text. Unverified references and unsupported claims are flagged.

Those checks have limits. Finding a quote does not prove that it supports the interpretation, and a search that finds nothing does not establish novelty. The brief is a starting point for reading and planning.

Together, these skills fit a straightforward workflow: check the literature before planning, implement the work, then request an independent code review.

Both repositories are open source, with installation instructions and examples in their READMEs.
