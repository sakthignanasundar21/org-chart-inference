# Development log (AI-assisted)
1. **Design first.** Chose truth-first generation over "make a CSV, then guess": the brief's hint about knowing if inference is right.
2. **First full run hung.** Cause: name generator looped forever once a company exceeded 400 people (only 400 first/last combinations). Fix: middle initial. Lesson: profile 40 companies before running 500.
3. **Raw accuracy was 44%.** Inspecting errors showed most were genuine ties between identical managers, not parser bugs, so the response was calibration and abstention rather than more rules.
4. **Failure gallery found a real bug.** ICs treated as "function head" when the head's title dropped its function word, producing false missing-leader ghosts. Restricted the rule to manager levels.
5. **Second gallery found another.** Managers with unknown function were never candidates. Added them as wildcard candidates: precision 88.4% to 91.0%, wrong-edge rate 3.5% to 2.5%.
6. **Typo vocabulary gap** ("Accunt Manager"): added "account" to the fuzzy-match vocabulary.
7. **Ablations were not all flattering.** Region signal does not help; ghost nodes cost a little precision. Reported as is.
8. **App** is one static file with 30 held-out companies embedded; smoke-tested in jsdom (tree, slider, truth overlay, Q&A).
9. **Asked "where does the parser fail?" instead of tuning the scorer.** Found that "Manager, Eng" got no function because only the first title segment was searched. One-line fix: coverage 28% to 37%, precision about 1 point lower.
10. **Team-to-function learned per company.** "Specialist I (Payments)" inherits Engineering because Payments is Engineering everywhere else in that company. Coverage 35.3%, precision 91.8%, ECE 0.0065.
11. **Ablation story flipped** after the fixes: global solver and ghost nodes now clearly help precision. Rewrote the write-up rather than keeping the old narrative.
12. **LLM tie-breaker built but not evaluated.** No API key in the sandbox. Verified plumbing with a stub and measured headroom (about 3.5 residual informative titles per company). Said so in the write-up instead of claiming a gain.
