"""Convert RTF calendar file to clean JSON."""
import re
import json

with open("pison-hack-gcal-demo.rtf", "r") as f:
    rtf = f.read()

# Extract lines after the RTF header
lines = rtf.split("\n")
json_lines = []
in_json = False
for line in lines:
    stripped = line.strip()
    if "\\cf0" in stripped:
        idx = stripped.index("\\cf0")
        stripped = stripped[idx + 5:]
        in_json = True
    if in_json:
        json_lines.append(stripped)

text = "\n".join(json_lines)

# Clean RTF escapes
text = text.replace("\\{", "{").replace("\\}", "}")

def replace_unicode(m):
    nums = [int(n) for n in re.findall(r"\\u(\d+)", m.group(0))]
    result = []
    i = 0
    while i < len(nums):
        n = nums[i]
        if 0xD800 <= n <= 0xDBFF and i + 1 < len(nums) and 0xDC00 <= nums[i + 1] <= 0xDFFF:
            hi, lo = n, nums[i + 1]
            codepoint = 0x10000 + (hi - 0xD800) * 0x400 + (lo - 0xDC00)
            result.append(chr(codepoint))
            i += 2
        else:
            try:
                result.append(chr(n))
            except (ValueError, OverflowError):
                result.append(f"U+{n:04X}")
                i += 1
                continue
            i += 1
    return "".join(result)

text = re.sub(r"\\uc0(?:\s*\\u\d+)+", replace_unicode, text)
text = re.sub(r"\\'([0-9a-fA-F]{2})", lambda m: chr(int(m.group(1), 16)), text)
text = text.replace("\\\\n", "\\n")
text = re.sub(r"\\\n", "\n", text)

text = text.rstrip()
depth = 0
end = 0
for i, c in enumerate(text):
    if c == "{":
        depth += 1
    elif c == "}":
        depth -= 1
        if depth == 0:
            end = i + 1
            break

text = text[:end]

data = json.loads(text)
print(f"Parsed {len(data['items'])} calendar events:")
for item in data["items"]:
    print(f"  - {item['summary']}: {item['start']['dateTime']}")

with open("hackathon/data/calendar.json", "w") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("\nSaved to hackathon/data/calendar.json")
