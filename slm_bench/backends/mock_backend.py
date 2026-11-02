"""
Mock inference backend for testing, dry-runs, and CI without loading real weights.
"""

import time
import re
import json
from typing import List, Dict, Any, Optional
from .base import BaseInferenceBackend, GenerationResult


class MockInferenceBackend(BaseInferenceBackend):
    """
    Simulated inference engine that produces realistic, context-aware answers
    for reasoning, code generation, instruction following, and tool calling.
    """

    def __init__(self, model_id: str, quantization: str = "BF16", simulated_latency_ms: float = 15.0, accuracy_bias: float = 0.95, **kwargs):
        super().__init__(model_id, quantization, **kwargs)
        self.simulated_latency_ms = simulated_latency_ms
        self.accuracy_bias = accuracy_bias

    def load(self) -> None:
        self.is_loaded = True

    def _generate_mock_response(self, text: str) -> str:
        prompt_lower = text.lower()

        # ─── Tool Calling (Catalog & Negative Tests) ──────────────────────────
        if "available tools" in prompt_lower:
            # Negative Tests: Must NOT return tool call
            if "photosynthesis converts sunlight" in prompt_lower:
                return "Photosynthesis is the biological process where chlorophyll pigments absorb solar photons, synthesizing ATP and NADPH to convert carbon dioxide into carbohydrates."
            if "capital city of australia" in prompt_lower:
                return "The capital city of Australia is Canberra."

            # Positive Tool Scenarios
            if "weather" in prompt_lower and "tokyo" in prompt_lower:
                return '{"name": "get_weather", "arguments": {"location": "Tokyo", "unit": "celsius"}}'
            if "25 + 75" in prompt_lower:
                return '{"name": "calculator", "arguments": {"operation": "add", "a": 25, "b": 75}}'
            if "model evaluation standards" in prompt_lower:
                return '{"name": "search_database", "arguments": {"query": "model evaluation standards"}}'
            if "/etc/nginx/nginx.conf" in prompt_lower or "nginx" in prompt_lower:
                return '{"name": "read_file", "arguments": {"filepath": "/etc/nginx/nginx.conf"}}'
            if "prod_db" in prompt_lower or "active users" in prompt_lower:
                return '{"name": "execute_sql", "arguments": {"query": "SELECT * FROM users WHERE active = 1", "database": "prod_db"}}'
            if "devops@company.com" in prompt_lower:
                return '{"name": "send_email", "arguments": {"recipient": "devops@company.com", "subject": "Disk Alert", "body": "Disk space is above 90%"}}'
            if "nvda" in prompt_lower:
                return '{"name": "get_stock_quote", "arguments": {"ticker": "NVDA"}}'
            if "architecture review" in prompt_lower:
                return '{"name": "create_calendar_event", "arguments": {"title": "Architecture Review", "start_time": "2025-06-01T14:00:00", "duration_minutes": 45}}'
            if "12 by 12" in prompt_lower:
                return '{"name": "calculator", "arguments": {"operation": "multiply", "a": 12, "b": 12}}'
            if "readme.md" in prompt_lower:
                return '{"name": "read_file", "arguments": {"filepath": "README.md"}}'

        # ─── ARC-Challenge Multiple Choice ────────────────────────────────────
        if "mineral" in prompt_lower and "scratch" in prompt_lower:
            return "Scratching a mineral with a copper penny tests Mohs hardness. #### C"
        if "solar oven" in prompt_lower and "thermal" in prompt_lower:
            return "Black surfaces absorb the most thermal radiation. #### C"
        if "rubber ball" in prompt_lower and "wall" in prompt_lower:
            return "When the ball pushes the wall, the wall exerts an equal and opposite force. #### C"
        if "phase change" in prompt_lower and "closer together" in prompt_lower:
            return "During condensation, gas molecules lose kinetic energy and turn into liquid. #### D"
        if "marine fish" in prompt_lower and "mountain" in prompt_lower:
            return "Marine fossils on mountain peaks indicate tectonic uplift from ancient ocean floors. #### B"
        if "plant cell" in prompt_lower and "chloroplast" in prompt_lower:
            return "Chloroplasts perform photosynthesis and are absent in animal cells. #### B"
        if "light bulb" in prompt_lower and "battery" in prompt_lower:
            return "The circuit converts electrical energy into radiant light and thermal heat. #### A"
        if "sound wave" in prompt_lower and "frequency" in prompt_lower:
            return "Higher frequency corresponds to higher perceived pitch. #### B"
        if "renewable" in prompt_lower and "resource" in prompt_lower:
            return "Solar energy is naturally replenished and renewable. #### C"
        if "tectonic plates" in prompt_lower and "transform fault" in prompt_lower:
            return "Transform fault boundaries frequently produce severe earthquakes. #### B"

        # ─── GSM-Symbolic Multi-Step Arithmetic ──────────────────────────────
        if "water tank" in prompt_lower and "liters" in prompt_lower:
            return "Net filling rate is 40 - 15 = 25 L/min. Time = 500 / 25 = 20 minutes. #### 20"
        if "train" in prompt_lower and "average speed" in prompt_lower:
            return "Distance = 75*2 + 90*3 = 150 + 270 = 420 km. Speed = 420 / 5 = 84 km/h. #### 84"
        if "chickens" in prompt_lower and "cows" in prompt_lower:
            return "Let c = cows and k = chickens. c + k = 30 and 4c + 2k = 88. 2c = 28, so c = 14. #### 14"
        if "laptop" in prompt_lower and "discount" in prompt_lower:
            return "$800 * 1.20 = $960. $960 * 0.80 = $768. #### 768"
        if "cookie" in prompt_lower and "batches" in prompt_lower:
            return "Total cookies = 15 * 24 = 360. Boxes = 360 / 8 = 45. Revenue = 45 * $6 = 270. #### 270"
        if "city a to city b" in prompt_lower or ("round trip" in prompt_lower and "speed" in prompt_lower):
            return "Time = 120/60 + 120/40 = 2 + 3 = 5 hours. Average speed = 240 / 5 = 48 mph. #### 48"
        if "consecutive integers" in prompt_lower and "72" in prompt_lower:
            return "3n = 72 => n = 24. The integers are 23, 24, 25. The largest is 25. #### 25"
        if "runner" in prompt_lower and "lap" in prompt_lower:
            return "Speed is 400m / 50s = 8 m/s. In 180 seconds, distance is 8 * 180 = 1440 meters. #### 1440"
        if "factory" in prompt_lower and "monday" in prompt_lower:
            return "Tuesday: 450 * 1.10 = 495. Wednesday: 495 - 50 = 445 items. #### 445"
        if "garden" in prompt_lower and "perimeter" in prompt_lower:
            return "2(w + w + 4) = 56 => 4w = 48 => w = 12, length = 16. Area = 12 * 16 = 192. #### 192"
        if "machines" in prompt_lower and "package" in prompt_lower:
            return "1 machine packages 2 boxes/min. 8 machines package 16 boxes/min * 15 min = 240. #### 240"
        if "cylinder" in prompt_lower and "capacity" in prompt_lower:
            return "180 / 0.75 = 240 ml total capacity. #### 240"
        if "alice" in prompt_lower and "bob" in prompt_lower:
            return "A = 2B. (A + 6) + (B + 6) = 48 => 3B + 12 = 48 => B = 12, A = 24. #### 24"
        if "apples" in prompt_lower and "oranges" in prompt_lower:
            return "Total apples = 40 + 20 = 60. Total fruit = 100 + 50 = 150. (60 / 150) * 100 = 40. #### 40"
        if "baker" in prompt_lower or "cakes" in prompt_lower:
            return "The baker makes 12 cakes per hour for 4 hours, which is 48 cakes. He sells 35, leaving 48 - 35 = 13 cakes. #### 13"

        # General Math Fallback
        if "calculate" in prompt_lower or "math" in prompt_lower or "solve" in prompt_lower:
            return "Step 1: Compute value. Step 2: Final calculation. #### 42"

        # ─── Code Generation / Agentic Coding ─────────────────────────────────
        if "fibonacci" in prompt_lower:
            return "```python\ndef fibonacci(n: int) -> int:\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n```"
        if "is_palindrome" in prompt_lower:
            return "```python\ndef is_palindrome(s: str) -> bool:\n    clean = ''.join(c.lower() for c in s if c.isalnum())\n    return clean == clean[::-1]\n```"
        if "flatten_list" in prompt_lower:
            return "```python\ndef flatten_list(nested: list) -> list:\n    res = []\n    for item in nested:\n        if isinstance(item, list):\n            res.extend(flatten_list(item))\n        else:\n            res.append(item)\n    return res\n```"
        if "two_sum" in prompt_lower:
            return "```python\ndef two_sum(nums: list[int], target: int) -> list[int]:\n    seen = {}\n    for i, n in enumerate(nums):\n        diff = target - n\n        if diff in seen:\n            return [seen[diff], i]\n        seen[n] = i\n    return []\n```"
        if "count_vowels" in prompt_lower:
            return "```python\ndef count_vowels(s: str) -> int:\n    return sum(1 for c in s.lower() if c in 'aeiou')\n```"
        if "find_longest_word" in prompt_lower:
            return "```python\ndef find_longest_word(sentence: str) -> str:\n    words = sentence.split()\n    return max(words, key=len) if words else ''\n```"
        if "is_valid_parentheses" in prompt_lower:
            return "```python\ndef is_valid_parentheses(s: str) -> bool:\n    stack = []\n    mapping = {')': '(', ']': '[', '}': '{'}\n    for char in s:\n        if char in mapping.values():\n            stack.append(char)\n        elif char in mapping:\n            if not stack or stack.pop() != mapping[char]:\n                return False\n    return len(stack) == 0\n```"
        if "find_missing_number" in prompt_lower:
            return "```python\ndef find_missing_number(nums: list[int]) -> int:\n    n = len(nums)\n    return n * (n + 1) // 2 - sum(nums)\n```"
        if "max_subarray_sum" in prompt_lower:
            return "```python\ndef max_subarray_sum(nums: list[int]) -> int:\n    max_so_far = current_max = nums[0]\n    for x in nums[1:]:\n        current_max = max(x, current_max + x)\n        max_so_far = max(max_so_far, current_max)\n    return max_so_far\n```"
        if "merge_intervals" in prompt_lower:
            return "```python\ndef merge_intervals(intervals: list[list[int]]) -> list[list[int]]:\n    if not intervals:\n        return []\n    intervals.sort(key=lambda x: x[0])\n    merged = [intervals[0]]\n    for current in intervals[1:]:\n        prev = merged[-1]\n        if current[0] <= prev[1]:\n            prev[1] = max(prev[1], current[1])\n        else:\n            merged.append(current)\n    return merged\n```"
        if "parse_ip_address" in prompt_lower:
            return "```python\ndef parse_ip_address(ip: str) -> bool:\n    parts = ip.split('.')\n    if len(parts) != 4:\n        return False\n    for p in parts:\n        if not p.isdigit():\n            return False\n        if len(p) > 1 and p.startswith('0'):\n            return False\n        if not (0 <= int(p) <= 255):\n            return False\n    return True\n```"
        if "extract_log_errors" in prompt_lower:
            return "```python\ndef extract_log_errors(logs: str) -> list[str]:\n    errors = []\n    for line in logs.split('\\n'):\n        if line.startswith('[ERROR] '):\n            errors.append(line[len('[ERROR] '):].strip())\n    return errors\n```"
        if "calculate_date_difference" in prompt_lower:
            return "```python\nimport datetime\ndef calculate_date_difference(d1: str, d2: str) -> int:\n    date1 = datetime.datetime.strptime(d1, '%Y-%m-%d')\n    date2 = datetime.datetime.strptime(d2, '%Y-%m-%d')\n    return abs((date2 - date1).days)\n```"
        if "compress_string" in prompt_lower:
            return "```python\ndef compress_string(s: str) -> str:\n    if not s:\n        return ''\n    res = []\n    count = 1\n    for i in range(1, len(s)):\n        if s[i] == s[i-1]:\n            count += 1\n        else:\n            res.append(s[i-1] + str(count))\n            count = 1\n    res.append(s[-1] + str(count))\n    compressed = ''.join(res)\n    return compressed if len(compressed) < len(s) else s\n```"
        if "longest_common_prefix" in prompt_lower:
            return "```python\ndef longest_common_prefix(strs: list[str]) -> str:\n    if not strs or not strs[0]:\n        return ''\n    prefix = strs[0]\n    for s in strs[1:]:\n        while not s.startswith(prefix):\n            prefix = prefix[:-1]\n            if not prefix:\n                return ''\n    return prefix\n```"
        if "is_power_of_two" in prompt_lower:
            return "```python\ndef is_power_of_two(n: int) -> bool:\n    return n > 0 and (n & (n - 1)) == 0\n```"
        if "reverse_words" in prompt_lower:
            return "```python\ndef reverse_words(s: str) -> str:\n    words = s.strip().split()\n    return ' '.join(reversed(words))\n```"
        if "top_k_frequent" in prompt_lower:
            return "```python\ndef top_k_frequent(nums: list[int], k: int) -> list[int]:\n    counts = {}\n    for n in nums:\n        counts[n] = counts.get(n, 0) + 1\n    sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)\n    return [item[0] for item in sorted_items[:k]]\n```"
        if "json_deep_find" in prompt_lower:
            return "```python\ndef json_deep_find(data: dict, target_key: str) -> list:\n    results = []\n    def search(obj):\n        if isinstance(obj, dict):\n            for k, v in obj.items():\n                if k == target_key:\n                    results.append(v)\n                search(v)\n        elif isinstance(obj, list):\n            for item in obj:\n                search(item)\n    search(data)\n    return results\n```"
        if "sliding_window_max" in prompt_lower:
            return "```python\ndef sliding_window_max(nums: list[int], k: int) -> list[int]:\n    if not nums or k <= 0:\n        return []\n    return [max(nums[i:i+k]) for i in range(len(nums) - k + 1)]\n```"

        # ─── Instruction Following (IFEval style) ─────────────────────────────
        if "france" in prompt_lower and "population_millions" in prompt_lower:
            return '{"country": "France", "capital": "Paris", "population_millions": 67.8}'
        if "without using the letter 'e'" in prompt_lower or "without using the letter e" in prompt_lower:
            return "A big brown fox jumps high in a sunny park."
        if "exactly 3 benefits" in prompt_lower or "exactly 3 bullet points" in prompt_lower:
            return "- Boosts cardiovascular health\n- Enhances muscular strength\n- Elevates mental clarity"
        if "between 20 and 30 words" in prompt_lower or "water cycle" in prompt_lower:
            return "Water evaporates from oceans into atmospheric clouds, condenses into heavy liquid droplets, falls down as precipitation over land, and flows back toward global oceans in continuous movement."
        if "certainly" in prompt_lower and "goodbye." in prompt_lower:
            return "Certainly, Python is an interpreted language executed by standard virtual machines. Goodbye."
        if "only lowercase" in prompt_lower:
            return "an atom is a fundamental unit of matter consisting of a nucleus surrounded by a cloud of orbiting electrons."
        if "all capital letters" in prompt_lower or "phishing" in prompt_lower:
            return "DO NOT OPEN UNVERIFIED EMAIL ATTACHMENTS OR PROVIDE PASSWORDS ON SUSPICIOUS LOGIN PORTALS."
        if "markdown table" in prompt_lower and "language" in prompt_lower:
            return "| Language | Paradigm | Typing |\n|---|---|---|\n| Python | Multi-paradigm | Dynamic |\n| Rust | Multi-paradigm | Static |"
        if "ai benchmark standard" in prompt_lower:
            return "The AI benchmark standard guarantees reproducible rigor. Adhering to this AI benchmark standard ensures reliable evaluations."
        if "without using any commas" in prompt_lower:
            return "The desert sands stretch endlessly under the blazing midday sun. Arid winds sculpt rolling dunes across the barren wasteland. Life persists through resilient cacti and nocturnal creatures."
        if "numbered list from 1. to 4." in prompt_lower:
            return "1. Spring\n2. Summer\n3. Autumn\n4. Winter"
        if "server configuration" in prompt_lower and "ssl" in prompt_lower:
            return '{"host": "api.domain.com", "port": 443, "ssl": true, "tags": ["production", "edge"]}'
        if "7, 42, and 100" in prompt_lower:
            return "Prime number 7 times 6 equals 42, while 100 represents a century."
        if "without using any of the following forbidden words" in prompt_lower:
            return "A motorized road transport vehicle carrying passengers on asphalt highways using petroleum fuel."
        if "exactly two paragraphs" in prompt_lower:
            return "Solar radiation generates clean electrical currents through modern photovoltaic silicon panels installed on residential rooftops.\n\nWind turbines capture atmospheric kinetic motion to spin electromagnetic generators supplying municipal utility grids."
        if "<analysis>" in prompt_lower:
            return "<analysis>Electric vehicles reduce urban emissions and enhance torque delivery across transit networks.</analysis>"
        if "[ref 1]" in prompt_lower:
            return "Objects accelerate toward Earth at approximately 9.8 meters per second squared due to gravitational attraction [Ref 1]."
        if "photosynthesis" in prompt_lower and "no bullet points" in prompt_lower:
            return "Photosynthesis is the biochemical mechanism by which green botanical organisms convert absorbed radiant sunlight, atmospheric carbon dioxide, and groundwater into glucose energy compounds and breathable oxygen."
        if "title case" in prompt_lower or "every single word begins with an uppercase letter" in prompt_lower:
            return "Every Good Bird Flies High"
        if "even number between 10 and 20" in prompt_lower or "night sky" in prompt_lower:
            return "Bright silver stars twinkle peacefully above the quiet sleepy world tonight."

        # ─── Tool Calling (Catalog & Negative Tests) ──────────────────────────
        # Negative Tests: Must NOT return tool call
        if "photosynthesis converts sunlight" in prompt_lower:
            return "Photosynthesis is the biological process where chlorophyll pigments absorb solar photons, synthesizing ATP and NADPH to convert carbon dioxide into carbohydrates."
        if "capital city of australia" in prompt_lower:
            return "The capital city of Australia is Canberra."

        # Positive Tool Scenarios
        if "weather" in prompt_lower and "tokyo" in prompt_lower:
            return '{"name": "get_weather", "arguments": {"location": "Tokyo", "unit": "celsius"}}'
        if "25 + 75" in prompt_lower:
            return '{"name": "calculator", "arguments": {"operation": "add", "a": 25, "b": 75}}'
        if "model evaluation standards" in prompt_lower:
            return '{"name": "search_database", "arguments": {"query": "model evaluation standards"}}'
        if "/etc/nginx/nginx.conf" in prompt_lower:
            return '{"name": "read_file", "arguments": {"filepath": "/etc/nginx/nginx.conf"}}'
        if "prod_db" in prompt_lower or "active users" in prompt_lower:
            return '{"name": "execute_sql", "arguments": {"query": "SELECT * FROM users WHERE active = 1", "database": "prod_db"}}'
        if "devops@company.com" in prompt_lower:
            return '{"name": "send_email", "arguments": {"recipient": "devops@company.com", "subject": "Disk Alert", "body": "Disk space is above 90%"}}'
        if "nvda" in prompt_lower:
            return '{"name": "get_stock_quote", "arguments": {"ticker": "NVDA"}}'
        if "architecture review" in prompt_lower:
            return '{"name": "create_calendar_event", "arguments": {"title": "Architecture Review", "start_time": "2025-06-01T14:00:00", "duration_minutes": 45}}'
        if "12 by 12" in prompt_lower:
            return '{"name": "calculator", "arguments": {"operation": "multiply", "a": 12, "b": 12}}'
        if "readme.md" in prompt_lower:
            return '{"name": "read_file", "arguments": {"filepath": "README.md"}}'

        return "This is a synthesized test response generated by the MockInferenceBackend."

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        stop_sequences: Optional[List[str]] = None,
    ) -> GenerationResult:
        t0 = time.perf_counter()
        if self.simulated_latency_ms > 0:
            time.sleep(self.simulated_latency_ms / 1000.0)
        output = self._generate_mock_response(prompt)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        tokens_est = len(output.split())

        if getattr(self, "on_stream_chunk", None):
            for word in output.split(" "):
                self.on_stream_chunk(word + " ")
        if getattr(self, "on_generation_result", None):
            self.on_generation_result({"text": output, "latency_ms": dt_ms, "tokens": tokens_est})

        return GenerationResult(output, dt_ms, tokens_est)

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        stop_sequences: Optional[List[str]] = None,
    ) -> GenerationResult:
        full_text = " ".join([m.get("content", "") for m in messages])
        return self.generate(full_text, max_new_tokens, temperature, stop_sequences)
