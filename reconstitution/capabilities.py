"""What a "capability" is, and the Python pack that instantiates one.

A capability is four things, and nothing about a model:

    tasks(stratum)   problems, graded by how much of the capability they need
    demos(stratum)   demonstration material to put in the context
    render(...)      how demos and a task become one prompt
    verify(...)      whether an answer is actually right

Point the harness at a different capability by writing another pack with
those four methods. Nothing in curve.py knows this one is Python.

Three design decisions worth arguing with
-----------------------------------------
**1. Strata, not one flat task set.** talkie's own report says every
correct vintage solution was "simple one-line programs ... or small
modifications to in-context example programs". A task set made only of real
HumanEval problems therefore returns all zeros, and all zeros has no curve
in it — you cannot tell "no recovery" from "recovery too small to see at
this n". So tasks come in three strata that bracket the reported ceiling:

    format-control   English pattern completion, no code at all
    oneline          one-expression function bodies
    humaneval-style  multi-line, real algorithmic content

`format-control` is the load-bearing one. If a model scores zero on it, the
zeros in the code strata say nothing about coding — they say the few-shot
channel itself did not work, which through a chat endpoint is entirely
possible. Any run that reports code zeros without a passing format control
is uninterpretable, and curve.py says so in the output.

**2. Dialects, to move the encoding penalty without moving the task.** The
same demonstrations rendered with tab indents, or with docstrings stripped,
or as comments, cost a different number of tokens while teaching exactly
the same thing. That is the only clean way to test whether the penalty
predicts recovery difficulty within a single model: everything else about
the experiment is held fixed and the token cost moves.

**3. Hand-written problems, NOT the HumanEval set.** Downloading HumanEval
would make these numbers superficially comparable to the report's 3.2%
pass@100 while differing in problem difficulty, sample count, and prompt
format. Hand-written and clearly labelled is the honest option. Pass
`--humaneval-jsonl` to run the real 164 problems when a genuinely
comparable number is the goal.
"""

import json
import re
from dataclasses import dataclass, field

import sandbox

CODE, EXACT = "code", "exact"


@dataclass(frozen=True)
class Task:
    id: str
    stratum: str
    stub: str            # the text the model continues
    check: str           # assert program (CODE) or accepted answers (EXACT)
    entry_point: str = ""
    kind: str = CODE


@dataclass(frozen=True)
class Demo:
    id: str
    source: str


@dataclass(frozen=True)
class Verdict:
    ok: bool
    reason: str
    candidate: str = ""


# --- demonstration pool: generic Python functions -----------------------

# Deliberately unrelated to the tasks, matching the report's "randomly
# chosen Python functions as examples to learn from in-context". One
# exception, flagged below.
_PY_DEMOS = [
    ("count_vowels", '''def count_vowels(text):
    """Return the number of vowels in text."""
    return sum(1 for ch in text.lower() if ch in "aeiou")
'''),
    ("add_one", '''def add_one(x):
    """Return x plus one."""
    return x + 1
'''),
    ("reverse_words", '''def reverse_words(sentence):
    """Return the words of sentence in reverse order."""
    return " ".join(reversed(sentence.split()))
'''),
    ("is_even", '''def is_even(n):
    """Return True when n is divisible by two."""
    return n % 2 == 0
'''),
    ("sum_list", '''def sum_list(numbers):
    """Return the total of every number in the list."""
    total = 0
    for n in numbers:
        total = total + n
    return total
'''),
    ("largest", '''def largest(items):
    """Return the greatest item in the list."""
    best = items[0]
    for item in items:
        if item > best:
            best = item
    return best
'''),
    ("repeat_string", '''def repeat_string(text, times):
    """Return text written out the given number of times."""
    return text * times
'''),
    ("celsius_to_fahrenheit", '''def celsius_to_fahrenheit(degrees):
    """Convert a temperature from centigrade to Fahrenheit."""
    return degrees * 9 / 5 + 32
'''),
    ("count_char", '''def count_char(text, letter):
    """Return how many times letter appears in text."""
    total = 0
    for ch in text:
        if ch == letter:
            total = total + 1
    return total
'''),
    ("strip_spaces", '''def strip_spaces(text):
    """Return text with the spaces at either end removed."""
    return text.strip()
'''),
    ("average", '''def average(numbers):
    """Return the mean of the numbers."""
    return sum(numbers) / len(numbers)
'''),
    ("initials", '''def initials(name):
    """Return the first letter of each word of a name."""
    return "".join(word[0] for word in name.split())
'''),
    ("only_positive", '''def only_positive(numbers):
    """Return a list of just the numbers greater than zero."""
    kept = []
    for n in numbers:
        if n > 0:
            kept.append(n)
    return kept
'''),
    ("shorter_of", '''def shorter_of(first, second):
    """Return whichever of the two strings is shorter."""
    if len(first) < len(second):
        return first
    return second
'''),
    ("join_with_dash", '''def join_with_dash(words):
    """Return the words joined by hyphens."""
    return "-".join(words)
'''),
    # The one non-random demo. talkie's report shows exactly one worked
    # success: given encode_shift in context, the model produced
    # decode_shift by turning a + into a -. Keeping this pair means the
    # harness can measure the report's own success case rather than only
    # problems it never claimed to solve.
    ("encode_shift", '''def encode_shift(s):
    """Return s with every letter moved five places up the alphabet."""
    return "".join([chr(((ord(ch) + 5 - ord("a")) % 26) + ord("a")) for ch in s])
'''),
]

# --- demonstration pool: English pattern pairs (format control) ---------

_EN_DEMOS = [
    ("plural-horse", "one horse, two horses"),
    ("plural-lamp", "one lamp, two lamps"),
    ("plural-street", "one street, two streets"),
    ("plural-window", "one window, two windows"),
    ("plural-letter", "one letter, two letters"),
    ("plural-garden", "one garden, two gardens"),
    ("plural-engine", "one engine, two engines"),
    ("plural-pocket", "one pocket, two pockets"),
    ("plural-market", "one market, two markets"),
    ("plural-bottle", "one bottle, two bottles"),
    ("plural-candle", "one candle, two candles"),
    ("plural-mirror", "one mirror, two mirrors"),
    ("plural-basket", "one basket, two baskets"),
    ("plural-button", "one button, two buttons"),
    ("plural-collar", "one collar, two collars"),
    ("plural-ribbon", "one ribbon, two ribbons"),
]


# --- tasks --------------------------------------------------------------

def _code_task(tid, stratum, entry, sig, doc, asserts):
    stub = f'def {entry}({sig}):\n    """{doc}"""\n'
    return Task(tid, stratum, stub, asserts, entry, CODE)


_ONELINE = [
    _code_task("add_two", "oneline", "add_two", "a, b",
               "Return the sum of a and b.",
               "assert add_two(2, 3) == 5\nassert add_two(-1, 1) == 0\n"
               "assert add_two(0, 0) == 0\n"),
    _code_task("subtract", "oneline", "subtract", "a, b",
               "Return a with b taken away.",
               "assert subtract(5, 3) == 2\nassert subtract(0, 4) == -4\n"),
    _code_task("double", "oneline", "double", "x",
               "Return twice x.",
               "assert double(4) == 8\nassert double(0) == 0\n"
               "assert double(-3) == -6\n"),
    _code_task("negate", "oneline", "negate", "x",
               "Return x with the opposite sign.",
               "assert negate(7) == -7\nassert negate(-2) == 2\n"),
    _code_task("last_item", "oneline", "last_item", "items",
               "Return the final item of the list.",
               "assert last_item([1, 2, 3]) == 3\n"
               "assert last_item(['a']) == 'a'\n"),
    _code_task("text_length", "oneline", "text_length", "text",
               "Return the number of letters in text.",
               "assert text_length('hello') == 5\nassert text_length('') == 0\n"),
    _code_task("to_upper", "oneline", "to_upper", "text",
               "Return text in capital letters.",
               "assert to_upper('abc') == 'ABC'\nassert to_upper('Hi') == 'HI'\n"),
    _code_task("multiply", "oneline", "multiply", "a, b",
               "Return a multiplied by b.",
               "assert multiply(3, 4) == 12\nassert multiply(5, 0) == 0\n"),
]

_HUMANEVAL_STYLE = [
    # The report's own success case: the inverse of a demonstrated function.
    _code_task("decode_shift", "humaneval-style", "decode_shift", "s",
               "Take a string encoded with encode_shift and return the "
               "original string.",
               "assert decode_shift('fghij') == 'abcde'\n"
               "assert decode_shift('mjqqt') == 'hello'\n"),
    _code_task("count_above", "humaneval-style", "count_above",
               "numbers, threshold",
               "Return how many of the numbers are greater than threshold.",
               "assert count_above([1, 5, 9], 4) == 2\n"
               "assert count_above([], 0) == 0\n"
               "assert count_above([2, 2, 2], 2) == 0\n"),
    _code_task("sum_even", "humaneval-style", "sum_even", "numbers",
               "Return the total of the even numbers in the list.",
               "assert sum_even([1, 2, 3, 4]) == 6\nassert sum_even([1, 3]) == 0\n"
               "assert sum_even([]) == 0\n"),
    _code_task("longest_word", "humaneval-style", "longest_word", "words",
               "Return the longest word in the list. If several are equally "
               "long, return the first of them.",
               "assert longest_word(['a', 'bbb', 'cc']) == 'bbb'\n"
               "assert longest_word(['ab', 'cd']) == 'ab'\n"),
    _code_task("is_palindrome", "humaneval-style", "is_palindrome", "text",
               "Return True when text reads the same backwards as forwards.",
               "assert is_palindrome('level') is True\n"
               "assert is_palindrome('lever') is False\n"
               "assert is_palindrome('') is True\n"),
    _code_task("unique_sorted", "humaneval-style", "unique_sorted", "items",
               "Return the list with repeats removed, in increasing order.",
               "assert unique_sorted([3, 1, 3, 2]) == [1, 2, 3]\n"
               "assert unique_sorted([]) == []\n"),
    _code_task("running_total", "humaneval-style", "running_total", "numbers",
               "Return a list whose each entry is the total of the numbers up "
               "to and including that place.",
               "assert running_total([1, 2, 3]) == [1, 3, 6]\n"
               "assert running_total([]) == []\n"),
    _code_task("words_starting", "humaneval-style", "words_starting",
               "words, letter",
               "Return the words that begin with the given letter.",
               "assert words_starting(['ant', 'bee', 'ape'], 'a') == "
               "['ant', 'ape']\n"
               "assert words_starting([], 'a') == []\n"),
]

_FORMAT_CONTROL = [
    Task("fc-clock", "format-control", "one clock, two", "clocks", "", EXACT),
    Task("fc-hammer", "format-control", "one hammer, two", "hammers", "", EXACT),
    Task("fc-saddle", "format-control", "one saddle, two", "saddles", "", EXACT),
    Task("fc-kettle", "format-control", "one kettle, two", "kettles", "", EXACT),
    Task("fc-lantern", "format-control", "one lantern, two", "lanterns", "", EXACT),
    Task("fc-parcel", "format-control", "one parcel, two", "parcels", "", EXACT),
    Task("fc-tunnel", "format-control", "one tunnel, two", "tunnels", "", EXACT),
    Task("fc-collar2", "format-control", "one satchel, two", "satchels", "", EXACT),
]


# --- dialects: same lesson, different token cost ------------------------

def _reindent(src, unit):
    """Replace each level of 4-space indent with `unit`. Semantics-preserving
    for these single-function snippets; changes only the token cost."""
    out = []
    for line in src.split("\n"):
        stripped = line.lstrip(" ")
        levels = (len(line) - len(stripped)) // 4
        out.append(unit * levels + stripped)
    return "\n".join(out)


_DOCSTRING = re.compile(r'^([ \t]*)"""([^"\n]*)"""[ \t]*\n', re.M)


def _strip_docstrings(src):
    return _DOCSTRING.sub("", src)


def _docstrings_to_comments(src):
    return _DOCSTRING.sub(
        lambda m: f"{m.group(1)}# {' '.join(m.group(2).split())}\n", src
    )


@dataclass(frozen=True)
class Dialect:
    """A surface form. `demo` rewrites the demonstration block; `stub`
    rewrites the task.

    They are separate on purpose. The task's docstring IS the problem
    statement, so a dialect that deletes docstrings must not touch it —
    otherwise the experiment stops holding the task fixed and starts asking
    the model to guess what the function is for, and the recovery
    difference gets misattributed to the token cost. Whitespace-only
    dialects apply to both, so the prompt is internally consistent.

    `content_preserving` marks the dialects that change ONLY how the same
    lesson is spelled — whitespace, nothing else. It decides which
    hypothesis a dialect can serve as evidence for, and getting this wrong
    is the easiest way to draw a false conclusion here:

      content-preserving (py-canonical, py-indent8, py-tab)
          the same demonstration, re-encoded. Differences in recovery are
          attributable to encoding cost, so these — and only these — test
          whether the TOKENIZER-DERIVED penalty predicts anything.

      content-changing (py-nodoc, py-comment)
          strictly less is taught, so cheaper demonstrations are also
          poorer ones. Useful for the mechanism (tokens per demonstration
          against recovery at a matched budget) and misleading for the
          proxy, because their low token cost comes from deleting material
          rather than from encoding it better.
    """

    demo: object
    stub: object
    content_preserving: bool = True


def _crlf(src):
    return src.replace("\r\n", "\n").replace("\n", "\r\n")


DIALECTS = {
    "py-canonical": Dialect(lambda s: s, lambda s: s),
    "py-nodoc": Dialect(_strip_docstrings, lambda s: s, content_preserving=False),
    "py-comment": Dialect(_docstrings_to_comments, lambda s: s,
                          content_preserving=False),
    "py-indent8": Dialect(lambda s: _reindent(s, " " * 8),
                          lambda s: _reindent(s, " " * 8)),
    "py-indent16": Dialect(lambda s: _reindent(s, " " * 16),
                           lambda s: _reindent(s, " " * 16)),
    "py-tab": Dialect(lambda s: _reindent(s, "\t"), lambda s: _reindent(s, "\t")),
    # Line endings are the purest encoding manipulation available: not one
    # character of content changes, and a byte-level BPE with no CRLF merges
    # pays for every line. If anything isolates "cost of spelling it out"
    # from "how much is being said", it is this one.
    "py-crlf": Dialect(_crlf, _crlf),
}

CONTENT_PRESERVING = tuple(k for k, v in DIALECTS.items() if v.content_preserving)


# --- extraction ---------------------------------------------------------

_FENCE = re.compile(r"```(?:[A-Za-z0-9_+-]*)\n(.*?)(?:```|\Z)", re.S)


def _normalise_indent(src):
    """Tabs to four spaces, CRLF to LF. A prompt in the py-tab dialect
    invites a tab-indented answer, and mixing that with the 4-space stub is
    a TabError that would be scored as a wrong answer rather than a
    formatting one; py-crlf invites carriage returns for the same reason."""
    return src.replace("\r\n", "\n").replace("\r", "\n").expandtabs(4)


_NEW_TOP_LEVEL = re.compile(r"^(def |class |@|import |from |if __name__)")


def _body_lines(text):
    """The leading run of lines that form a function body, however indented.

    Two channel behaviours have to be handled, and the difference is not
    cosmetic — get it wrong and every correct answer is scored as a
    non-answer:

      indented   a base-model continuation resumes inside the block, so the
                 body arrives at column 4 and the first column-0 line is
                 where the model moved on.
      column 0   the hosted chat endpoint returns 'return a + b' flush left
                 even though the prompt ends mid-block. Here column-0 lines
                 ARE the body, and only a new top-level definition ends it.

    Which regime we are in is decided by the first non-blank line, then held
    for the rest of the block.
    """
    lines = text.split("\n")
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    lines = lines[i:]
    if not lines:
        return ""
    flush_left = lines[0][:1] not in (" ", "\t")
    kept = []
    for line in lines:
        if not line.strip() or line[0] in " \t":
            kept.append(line)
            continue
        if not kept:
            kept.append(line)
            continue
        if flush_left and not _NEW_TOP_LEVEL.match(line):
            kept.append(line)
            continue
        break
    while kept and not kept[-1].strip():
        kept.pop()
    return "\n".join(kept)


def _indent_body(body, width=4):
    """Shift a body so its outermost level sits at `width`, keeping relative
    structure. A no-op when the body is already indented."""
    real = [ln for ln in body.split("\n") if ln.strip()]
    if not real:
        return body
    base = min(len(ln) - len(ln.lstrip(" ")) for ln in real)
    if base >= width:
        return body
    pad = " " * (width - base)
    return "\n".join(pad + ln if ln.strip() else ln for ln in body.split("\n"))


def _compiles(src):
    try:
        compile(src, "<candidate>", "exec")
    except (SyntaxError, ValueError):
        return False
    return True


# --- the pack -----------------------------------------------------------

@dataclass
class PythonCapability:
    """Writing correct Python from in-context demonstrations."""

    name: str = "python"
    strata: tuple = ("format-control", "oneline", "humaneval-style")
    dialects: tuple = tuple(DIALECTS)
    extra_tasks: list = field(default_factory=list)

    # -- data
    def tasks(self, stratum):
        if stratum == "oneline":
            return list(_ONELINE)
        if stratum == "humaneval-style":
            return list(_HUMANEVAL_STYLE) + list(self.extra_tasks)
        if stratum == "format-control":
            return list(_FORMAT_CONTROL)
        raise ValueError(f"unknown stratum {stratum!r}; have {self.strata}")

    def demos(self, stratum):
        pool = _EN_DEMOS if stratum == "format-control" else _PY_DEMOS
        return [Demo(i, s) for i, s in pool]

    def is_code(self, stratum):
        return stratum != "format-control"

    def dialect_of(self, name):
        return DIALECTS[name]

    # -- prompt
    def render(self, task, demos, dialect="py-canonical"):
        """Few-shot prompt: worked examples, then the task, cut off exactly
        where the answer begins. No instruction line — a bare continuation
        is what a base model wants, and phrasing an instruction would import
        modern assistant idiom into a test about pre-1931 text. The chat
        backend's own wrapper, if any, is recorded separately."""
        if task.kind == EXACT:
            shown = "\n".join(d.source for d in demos)
            return (shown + "\n" + task.stub) if shown else task.stub
        dl = DIALECTS[dialect]
        blocks = [dl.demo(d.source).rstrip("\n") for d in demos]
        blocks.append(dl.stub(task.stub).rstrip("\n") + "\n")
        return "\n\n".join(blocks)

    def penalty_material(self, dialect="py-canonical", n=8):
        """The bytes whose encoding cost is the predictor: the demonstration
        block itself, in the dialect actually used."""
        dl = DIALECTS[dialect]
        return "\n\n".join(dl.demo(s).rstrip("\n") for _, s in _PY_DEMOS[:n]) + "\n"

    # -- grading
    def extract(self, completion, task):
        """Recover an answer from a completion, trying the cheapest reading
        first. Returns (candidate_program, strategy). The strategy is
        recorded per generation: a run where most answers only survived the
        loosest reading is measuring extraction, not capability."""
        if task.kind == EXACT:
            first = completion.strip().split("\n")[0]
            word = re.sub(r"^[^A-Za-z]*", "", first)
            word = re.split(r"[^A-Za-z]", word)[0] if word else ""
            return word.lower(), "first-word"

        text = _normalise_indent(completion)
        stub = _normalise_indent(task.stub)
        attempts = []

        m = _FENCE.search(text)
        if m:
            attempts.append(("fence", m.group(1)))
        if f"def {task.entry_point}" in text:
            cut = text[text.index(f"def {task.entry_point}"):]
            attempts.append(("whole-def", cut))
        body = _body_lines(text)
        if body.strip():
            attempts.append(("body", stub + body))
            reindented = _indent_body(body)
            if reindented != body:
                attempts.append(("body-reindent", stub + reindented))
        if not attempts:
            # Nothing here is an answer. The stub alone compiles (a function
            # whose body is just its docstring) and would then be graded as
            # a wrong answer — but it is not wrong, it is absent, and the
            # two have to be counted apart or a mute model reads as an
            # incorrect one.
            return stub, "no-answer"
        attempts.append(("raw", stub + text))

        for strategy, cand in attempts:
            # A fence or whole-def reading may or may not already contain the
            # signature; add it when it does not.
            prog = cand if f"def {task.entry_point}" in cand else stub + cand
            prog = _normalise_indent(prog)
            if _compiles(prog) and f"def {task.entry_point}" in prog:
                return prog, strategy
        return stub + text, "uncompilable"

    def verify(self, task, candidate, **kw):
        if task.kind == EXACT:
            ok = candidate.strip().lower() == task.check.strip().lower()
            return Verdict(ok, "" if ok else f"got {candidate!r}", candidate=candidate)
        if not _compiles(candidate):
            return Verdict(False, "syntax error", candidate=candidate)
        res = sandbox.run_program(candidate + "\n" + task.check, **kw)
        return Verdict(res.ok, res.failure, candidate=candidate)


def load_humaneval(path):
    """Real HumanEval problems, for a number comparable to the report's.

    Expects the official JSONL (prompt / test / entry_point). Not shipped:
    fetch it from github.com/openai/human-eval. Using this changes what the
    numbers mean, so curve.py records the path in the results file.
    """
    tasks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            check = (rec["test"] + "\n"
                     f"check({rec['entry_point']})\n")
            tasks.append(Task(rec["task_id"], "humaneval", rec["prompt"],
                              check, rec["entry_point"], CODE))
    return tasks


REGISTRY = {"python": PythonCapability}


def load_capability(name, humaneval_jsonl=None):
    cap = REGISTRY[name]()
    if humaneval_jsonl:
        cap.extra_tasks = load_humaneval(humaneval_jsonl)
    return cap
