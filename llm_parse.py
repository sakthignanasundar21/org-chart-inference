"""Optional LLM tie-breaker: infer the FUNCTION of titles the rules and the company's own team vocabulary could not resolve.
Cached on disk, batched, deterministic (temperature 0), and only ever called for unresolved unique titles.
Needs ANTHROPIC_API_KEY. Not exercised in this repo's published numbers (no key in the build sandbox)."""
import os, json, hashlib, urllib.request

FUNCS = ["Engineering", "Data", "Security", "Product", "Design", "Sales", "Marketing", "Finance", "People", "Operations", "Legal", "Support"]

def make_llm_parser(cache_path="llm_cache.json", model="claude-haiku-5-5", batch=40):
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    def call(prompt):
        req = urllib.request.Request("https://api.anthropic.com/v1/messages", method="POST",
            headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01", "content-type": "application/json"},
            data=json.dumps({"model": model, "max_tokens": 2000, "temperature": 0, "messages": [{"role": "user", "content": prompt}]}).encode())
        return json.load(urllib.request.urlopen(req, timeout=60))["content"][0]["text"]
    def parse(titles, teams):
        key = lambda t: hashlib.md5(t.encode()).hexdigest()
        todo = [t for t in titles if key(t) not in cache]
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            prompt = ("For each job title, name the company function it belongs to, choosing ONLY from " + ", ".join(FUNCS) +
                      ". Team names seen at this company: " + ", ".join(teams) + ". If the title carries no usable evidence answer null; never guess. "
                      "Reply with ONLY a JSON object mapping each title string to a function or null.\n" + json.dumps(chunk))
            try: out = json.loads(call(prompt))
            except Exception: out = {}
            for t in chunk: cache[key(t)] = out.get(t) if out.get(t) in FUNCS else None
            json.dump(cache, open(cache_path, "w"))
        return {t: cache.get(key(t)) for t in titles}
    return parse
