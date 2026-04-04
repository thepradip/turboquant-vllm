#!/usr/bin/env python3
"""
Full Report Benchmark: Bonsai-8B across contexts and KV configs.
20 QA questions x 4 KV configs x multiple context sizes.
Measures: quality (pass/fail), TTFT (pp tok/s), gen tok/s, wall time.
"""

import subprocess, os, re, time, json, sys

BONSAI_DIR = "/Users/pradip/Desktop/Learning/Claude/PrismML/Bonsai-demo"
LLAMA_CLI = os.path.join(BONSAI_DIR, "bin", "mac", "llama-cli")
DYLD = os.path.join(BONSAI_DIR, "bin", "mac")
MODEL = os.path.join(BONSAI_DIR, "models", "Bonsai-8B.gguf")

KV_CONFIGS = [
    {"name": "f16",  "ctk": "f16",  "ctv": "f16"},
    {"name": "q8_0", "ctk": "q8_0", "ctv": "q8_0"},
    {"name": "q4_0", "ctk": "q4_0", "ctv": "q4_0"},
    {"name": "k8v4", "ctk": "q8_0", "ctv": "q4_0"},
]

CONTEXTS = [1024, 2048, 4096, 8192, 16384, 32768]

QA = [
    {"id":1, "cat":"RAG", "q":'Based on this passage, answer the question.\n\nPassage: "Meridian Technologies reported Q3 2024 revenue of $4.2 billion, up 18% year-over-year. The growth was primarily driven by their cloud infrastructure division, which grew 34% to $1.8 billion. However, their legacy hardware segment declined 12% to $890 million. CEO Sarah Chen noted that operating margins improved to 22.5%, up from 19.1% in Q3 2023. The company revised its full-year guidance upward to $17.2 billion, from the previous estimate of $16.5 billion."\n\nQuestion: What was the dollar amount of revenue growth in the cloud infrastructure division compared to the same quarter last year? Show your calculation.',
     "check": lambda a: ("1.34" in a or "340" in a or "1342" in a or "1.8" in a) and ("34" in a)},
    {"id":2, "cat":"RAG", "q":'Read the following context and answer precisely.\n\nContext: "Project Aurora has three phases. Phase 1 (completed Jan 2024) migrated 2,400 user accounts from SystemA to SystemB. Phase 2 (completed Apr 2024) migrated 1,800 accounts from SystemC to SystemB. Phase 3 (planned for Jul 2024) will migrate the remaining accounts from SystemD. The total number of accounts across all four legacy systems was 6,100 before migration began. SystemB had 500 existing accounts before any migration started."\n\nQuestions:\n1. How many accounts will Phase 3 migrate?\n2. How many total accounts will SystemB have after all phases complete?',
     "check": lambda a: ("1900" in a or "1,900" in a) and ("6600" in a or "6,600" in a or "6100" in a.replace(",",""))},
    {"id":3, "cat":"RAG", "q":'Read these two passages and identify the contradiction.\n\nPassage A (Internal Memo, March 5): "The new authentication system will use OAuth 2.0 with PKCE flow. All API endpoints must migrate by April 30. The security team confirmed that legacy API keys will be deprecated and disabled on May 1."\n\nPassage B (Client Communication, March 12): "We want to assure all partners that your existing API key integrations will continue to work throughout 2024. No action is required on your part at this time."\n\nWhat is the specific contradiction between these two documents?',
     "check": lambda a: ("api key" in a.lower()) and ("contradict" in a.lower() or "conflict" in a.lower() or "inconsisten" in a.lower() or "deprecated" in a.lower())},
    {"id":4, "cat":"RAG", "q":'Based ONLY on the provided context, answer the question. If the answer is not in the context, say "NOT FOUND IN CONTEXT".\n\nContext: "Acme Corp\'s data center in Oregon uses 45MW of power and has a PUE ratio of 1.3. The facility was built in 2019 and houses approximately 50,000 servers."\n\nQuestion: What is Acme Corp\'s annual electricity cost for the Oregon data center?',
     "check": lambda a: "not found" in a.lower() or "not in the context" in a.lower() or "not mentioned" in a.lower() or "cannot be determined" in a.lower() or "not provided" in a.lower()},
    {"id":5, "cat":"RAG", "q":'Summarize the following in EXACTLY 3 bullet points. Each bullet must be one sentence only.\n\nText: "The European Union\'s AI Act establishes a risk-based regulatory framework for AI systems. High-risk AI applications in healthcare, law enforcement, and critical infrastructure face mandatory conformity assessments and transparency obligations. General-purpose AI models must comply with transparency requirements and copyright rules. Systems deemed to pose unacceptable risk, such as social scoring and real-time biometric surveillance, are banned. Companies face fines of up to 7% of global revenue for violations."',
     "check": lambda a: a.count("- ") >= 2 or a.count("* ") >= 2 or a.count("1.") >= 1},
    {"id":6, "cat":"Finance", "q":'From the following financial data, calculate the debt-to-equity ratio and current ratio. Show your work.\n\nBalance Sheet (in millions):\n- Total Assets: $12,400\n- Current Assets: $3,200\n- Total Liabilities: $7,800\n- Current Liabilities: $2,100\n- Shareholders\' Equity: $4,600\n- Long-term Debt: $5,700',
     "check": lambda a: ("1.7" in a or "1.69" in a) and ("1.52" in a or "1.5" in a)},
    {"id":7, "cat":"Finance", "q":'Read this earnings call excerpt and answer: Is the company\'s outlook positive, negative, or mixed? Cite specific phrases.\n\n"While we delivered solid top-line growth of 12%, we want to be transparent about headwinds. Gross margins compressed 180 basis points due to elevated input costs, and we expect this pressure to persist through Q1 2025. On the positive side, our subscription revenue reached an inflection point, now representing 62% of total revenue versus 41% a year ago. We\'re cautiously optimistic about the second half but have prudently adjusted our full-year EPS guidance down by $0.15."',
     "check": lambda a: "mixed" in a.lower() and ("margin" in a.lower() or "headwind" in a.lower()) and ("subscription" in a.lower() or "growth" in a.lower())},
    {"id":8, "cat":"Finance", "q":'Extract these fields from the invoice: invoice_number, date, vendor_name, vendor_id, client_name, subtotal, tax_rate, total, payment_terms, po_reference\n\n"Invoice #INV-2024-0847 dated November 15, 2024 from CloudServe Inc. (vendor ID: VS-1192) to Pinnacle Dynamics LLC for Professional Services. Subtotal: $34,500.00. Tax (8.5%): $2,932.50. Total Due: $37,432.50. Payment terms: Net 30. PO Reference: PO-2024-445."',
     "check": lambda a: "INV-2024-0847" in a and "VS-1192" in a and "37,432.50" in a and "PO-2024-445" in a},
    {"id":9, "cat":"Finance", "q":'Analyze the revenue trend:\n\nQuarterly Revenue (in $M):\nQ1 2023: 120, Q2 2023: 135, Q3 2023: 128, Q4 2023: 152\nQ1 2024: 141, Q2 2024: 158, Q3 2024: 149, Q4 2024: 175\n\n1. What is the year-over-year growth rate for Q4?\n2. Which quarter shows the weakest sequential growth in 2024?\n3. Is there a seasonal pattern?',
     "check": lambda a: ("15" in a and "%" in a) and ("q3" in a.lower() or "q1" in a.lower())},
    {"id":10, "cat":"Finance", "q":'Based on this risk disclosure, list the top 3 risks in order of severity and explain your ranking.\n\n"The Company faces several material risks: (1) Concentration risk - top 3 clients represent 68% of revenue; (2) Regulatory risk - pending EU legislation could require $12M in product modifications; (3) Currency risk - 40% of revenue in foreign currencies, unhedged; (4) Key person risk - CTO holds 14 critical patents; (5) Cybersecurity risk - two minor data incidents in 2024, contained."',
     "check": lambda a: "concentration" in a.lower() and ("68%" in a or "top 3" in a.lower())},
    {"id":11, "cat":"Reasoning", "q":'Solve step by step: Five people (Alice, Bob, Carol, Dave, Eve) sit in chairs 1-5 (left to right).\n- Bob sits immediately to the right of Alice.\n- Carol is not in chair 1 or 5.\n- Dave sits in chair 4.\n- Eve is not adjacent to Dave.\nWhere does each person sit?',
     "check": lambda a: "alice" in a.lower() and "bob" in a.lower() and "dave" in a.lower() and "4" in a},
    {"id":12, "cat":"Reasoning", "q":'A server went down at 3:00 AM. Timeline:\n- 2:45 AM: Deployment pushed version 2.3.1\n- 2:50 AM: Memory usage spiked 60% to 95%\n- 2:55 AM: DB connection pool exhausted (max 100)\n- 2:58 AM: Error rate jumped 0.1% to 45%\n- 3:00 AM: Health check failed, server removed\n- 3:02 AM: Rollback to 2.3.0 initiated\n- 3:08 AM: Memory dropped to 65%, connections normalized\n\nWhat is the root cause? What evidence supports this? How to prevent recurrence?',
     "check": lambda a: ("2.3.1" in a or "deploy" in a.lower()) and ("memory" in a.lower() or "leak" in a.lower())},
    {"id":13, "cat":"Reasoning", "q":'Evaluate each conclusion as VALID or INVALID based strictly on the premises.\n\nPremises:\n1. All ML engineers know Python.\n2. Some Python developers are not ML engineers.\n3. No data analyst at this company knows R.\n4. All data analysts at this company know Python.\n\nConclusions:\nA. "Some Python developers are data analysts." - VALID or INVALID?\nB. "No ML engineer knows R." - VALID or INVALID?\nC. "All people who know Python are ML engineers." - VALID or INVALID?',
     "check": lambda a: "invalid" in a.lower() and a.lower().count("invalid") >= 2},
    {"id":14, "cat":"Reasoning", "q":'A company has $500,000 annual cloud budget split across 3 providers. Constraints:\n- AWS must get at least 40% ($200K)\n- GCP cannot get more than $150,000\n- Azure must get at least $50K more than GCP\n- Total must equal exactly $500,000\n\nFind a valid allocation. What is the maximum possible amount for GCP?',
     "check": lambda a: ("200" in a) and ("150" in a or "100" in a)},
    {"id":15, "cat":"Reasoning", "q":'Complete each analogy and explain in one sentence:\n1. Docker container is to application as _____ is to virtual machine.\n2. Git branch is to codebase as _____ is to database.\n3. Load balancer is to servers as _____ is to network traffic.\n4. API rate limiting is to server resources as _____ is to highway traffic.',
     "check": lambda a: len(a) > 100 and ("hypervisor" in a.lower() or "image" in a.lower() or "snapshot" in a.lower() or "os" in a.lower() or "host" in a.lower())},
    {"id":16, "cat":"Instruction", "q":'Convert this paragraph into a JSON object with keys: "title", "date", "amount", "currency", "status". Respond with ONLY the JSON.\n\n"The payment of 2,500 euros for the Q3 Marketing Campaign was approved on September 15, 2024 and is currently pending disbursement."',
     "check": lambda a: '"title"' in a and '"date"' in a and '"amount"' in a and "{" in a and "}" in a},
    {"id":17, "cat":"Instruction", "q":'Process these tickets according to the rules:\n\nRules:\n- severity=critical AND status=open: assign Tier 3, priority P0\n- severity=high AND created >24h ago: assign Tier 2, priority P1\n- All other: assign Tier 1, priority P2\n\nTickets:\n1. T-001: critical, open, 2h ago\n2. T-002: high, open, 36h ago\n3. T-003: medium, open, 1h ago\n4. T-004: high, open, 12h ago\n5. T-005: critical, resolved, 48h ago\n\nList each ticket assignment and priority.',
     "check": lambda a: "T-001" in a and ("Tier 3" in a or "tier 3" in a.lower()) and ("P0" in a) and ("Tier 2" in a or "tier 2" in a.lower())},
    {"id":18, "cat":"Instruction", "q":'Answer about Python decorators. Rules:\n- Do NOT use code blocks or backticks\n- Do NOT mention the word "function"\n- Do NOT use bullet points or numbered lists\n- Answer in exactly 2 sentences',
     "check": lambda a: "```" not in a and "function" not in a.lower() and len(a.split("\n")) <= 5},
    {"id":19, "cat":"Instruction", "q":'Review this code and identify ALL bugs. For each: (a) the line, (b) what is wrong, (c) the fix.\n\ndef calculate_average(numbers):\n    total = 0\n    for i in range(len(numbers)):\n        total += numbers[i]\n    average = total / len(numbers)\n    return average\n\nresult = calculate_average([])\nprint(f"Average: {result}")',
     "check": lambda a: ("division" in a.lower() or "zero" in a.lower() or "empty" in a.lower() or "ZeroDivision" in a)},
    {"id":20, "cat":"Instruction", "q":'Write a SQL query. Schema: orders(id, customer_id, amount, status, created_at), customers(id, name, email, tier).\n\nRequest: "Find premium tier customers with more than 5 orders in the last 90 days whose total amount is under $1000. Show name, order count, total amount. Sort by total desc. Exclude cancelled orders."',
     "check": lambda a: "select" in a.lower() and "join" in a.lower() and "group by" in a.lower() and "having" in a.lower() and "premium" in a.lower()},
]


def run(question, ctk, ctv, ctx, max_tok=400):
    prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    cmd = [
        LLAMA_CLI, "-m", MODEL, "-ngl", "999", "-fa", "1",
        "-c", str(ctx), "-ctk", ctk, "-ctv", ctv,
        "-n", str(max_tok), "-p", prompt,
        "--temp", "0", "--single-turn", "--no-display-prompt",
    ]
    env = os.environ.copy()
    env["DYLD_LIBRARY_PATH"] = DYLD
    start = time.perf_counter()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
    except subprocess.TimeoutExpired:
        return "[TIMEOUT]", 0, 0, time.perf_counter() - start
    wall = time.perf_counter() - start
    full = r.stdout + "\n" + r.stderr
    m = re.search(r"Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s", full)
    pp = float(m.group(1)) if m else 0
    gen = float(m.group(2)) if m else 0
    ans = r.stdout
    for noise in ["Loading model...", "Exiting..."]:
        ans = ans.replace(noise, "")
    ans = re.sub(r"[▄▀█░▓▒]+[^\n]*", "", ans)
    ans = re.sub(r"\x1b\[[0-9;]*m", "", ans)
    ans = re.sub(r"\[ Prompt:.*?\]", "", ans)
    lines = [re.sub(r"^[\|\\\-/]+\s*", "", l.strip()) for l in ans.split("\n")
             if l.strip() and not any(x in l for x in [">","<|im","build ","model ","modalities","available","exit","regen","clear","/read"])]
    return "\n".join(lines).strip(), pp, gen, wall


def main():
    # Run 20 questions at context=8192 for all 4 KV configs (main quality test)
    ctx = 8192
    print(f"{'='*80}")
    print(f"  Bonsai-8B -- 20 Production QA at ctx={ctx}")
    print(f"{'='*80}")

    all_scores = {}
    all_metrics = {}

    for kv in KV_CONFIGS:
        passed = 0
        total_wall = 0
        total_pp = 0
        total_gen = 0
        n = 0
        for q in QA:
            ans, pp, gen, wall = run(q["q"], kv["ctk"], kv["ctv"], ctx)
            ok = q["check"](ans)
            if ok: passed += 1
            total_wall += wall
            if pp > 0:
                total_pp += pp; total_gen += gen; n += 1
            tag = "OK" if ok else "X "
            print(f"  [{kv['name']:<4}] Q{q['id']:>2} {q['cat']:<11} [{tag}] {wall:>5.1f}s pp={pp:.0f} gen={gen:.0f}")

        avg_pp = total_pp / max(n, 1)
        avg_gen = total_gen / max(n, 1)
        all_scores[kv["name"]] = passed
        all_metrics[kv["name"]] = {"wall": total_wall, "pp": avg_pp, "gen": avg_gen}
        print(f"  [{kv['name']}] Score: {passed}/20 | Wall: {total_wall:.0f}s | pp: {avg_pp:.0f} | gen: {avg_gen:.0f}")

    # Context scaling test: run 5 questions at each context size with f16 and q4_0
    print(f"\n{'='*80}")
    print(f"  Context Scaling: f16 vs q4_0 (5 questions)")
    print(f"{'='*80}")
    print(f"  {'Ctx':>6} {'f16 Score':>10} {'q4_0 Score':>11} {'f16 pp':>8} {'q4_0 pp':>8} {'f16 gen':>8} {'q4_0 gen':>9}")
    print(f"  {'-'*70}")

    ctx_results = {}
    for ctx in CONTEXTS:
        for kv_name, ctk, ctv in [("f16", "f16", "f16"), ("q4_0", "q4_0", "q4_0")]:
            passed = 0; sum_pp = 0; sum_gen = 0; sum_wall = 0; n = 0
            for q in QA[:5]:  # First 5 RAG questions
                ans, pp, gen, wall = run(q["q"], ctk, ctv, ctx)
                if q["check"](ans): passed += 1
                sum_wall += wall
                if pp > 0: sum_pp += pp; sum_gen += gen; n += 1
            ctx_results[(ctx, kv_name)] = {
                "score": passed, "pp": sum_pp/max(n,1), "gen": sum_gen/max(n,1), "wall": sum_wall
            }

        f = ctx_results[(ctx, "f16")]
        q = ctx_results[(ctx, "q4_0")]
        print(f"  {ctx:>6} {f['score']:>7}/5   {q['score']:>7}/5   {f['pp']:>7.0f} {q['pp']:>8.0f} {f['gen']:>8.0f} {q['gen']:>9.0f}")

    # Summary
    print(f"\n{'='*80}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"\n  Quality (20 questions, ctx=8192):")
    print(f"  {'Config':<25} {'Score':>7} {'vs FP16':>8} {'Wall':>7} {'PP tok/s':>9} {'Gen tok/s':>10}")
    print(f"  {'-'*68}")
    base = all_scores["f16"]
    for kv in KV_CONFIGS:
        s = all_scores[kv["name"]]
        m = all_metrics[kv["name"]]
        d = f"{s-base:+d}" if kv["name"] != "f16" else "---"
        print(f"  {kv['name']:<25} {s:>5}/20 {d:>8} {m['wall']:>6.0f}s {m['pp']:>9.0f} {m['gen']:>10.0f}")

    # Save
    out = {
        "model": "Bonsai-8B", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "quality_scores": all_scores, "metrics": {k: {kk: round(vv,2) for kk,vv in v.items()} for k,v in all_metrics.items()},
        "context_scaling": {f"{ctx}_{kv}": {kk: round(vv,2) for kk,vv in v.items()} for (ctx,kv),v in ctx_results.items()},
    }
    with open("full_report_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved to full_report_results.json")


if __name__ == "__main__":
    main()
