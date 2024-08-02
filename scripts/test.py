import unittest
import subprocess
import os
import re

OPENCV = os.environ["OPENCV"]
BUILD = os.environ["BUILD"]

class MatchResult:
    def __init__(self, arr):
        self.baseline = arr[0]
        self.requested = arr[1]
        self.disabled = arr[2]
        self.dispatched = arr[3]
    def __repr__(self):
        return "({}, {}, {}, {})".format(self.baseline, self.requested, self.disabled, self.dispatched)

def parse_output(stdout):
    def get_group(match, idx=1):
        return match.group(idx) if match is not None else None
    
    begin = re.search(r"CPU/HW features:", stdout).end(0)
    end = re.search(r"C/C\+\+:", stdout).start(0)
    sub = stdout[begin:end]
    return MatchResult([
        get_group(re.search(r"^--\s+Baseline:\s+(.*)$", sub, re.MULTILINE)), 
        get_group(re.search(r"^--\s+requested:\s+(.*)$", sub, re.MULTILINE)), 
        get_group(re.search(r"^--\s+disabled:\s+(.*)$", sub, re.MULTILINE)), 
        get_group(re.search(r"^--\s+Dispatched code generation:(\n--\s+requested:)?\s+(.*)$", sub, re.MULTILINE), 2)
    ]), sub


class TestBase(unittest.TestCase):
    def check_features(self, popen_res, expected):
        self.assertEqual(popen_res.returncode, 0)
        actual, raw = parse_output(popen_res.stdout.decode("utf-8"))
        self.assertDictEqual(actual.__dict__, expected.__dict__, "\nResults are different, actual raw output:\n{}\n".format(raw))

    def setUp(self):
        subprocess.run("rm -rf *", shell=True, cwd=BUILD)


class Test_x86_64(TestBase):
    def run_configure(self, args):
        return subprocess.run(["cmake", "-GNinja"] + args + [OPENCV], capture_output=True, cwd=BUILD)

    def test_default(self):
        res = self.run_configure([])
        self.check_features(res, MatchResult(["SSE SSE2 SSE3", "SSE3", None, "SSE4_1 SSE4_2 AVX FP16 AVX2 AVX512_SKX"]))

    def test_disable_sse2(self):
        res = self.run_configure(["-DCPU_BASELINE_DISABLE=SSE2"])
        self.check_features(res, MatchResult(["SSE", "SSE3", "SSE2", "SSE4_1 SSE4_2 AVX FP16 AVX2 AVX512_SKX"]))

    def test_detect(self):
        res = self.run_configure(["-DCPU_BASELINE=DETECT"])
        self.check_features(res, MatchResult(["SSE SSE2", "DETECT", None, "SSE4_1 SSE4_2 AVX FP16 AVX2 AVX512_SKX"]))

    # will fail on other platforms than specific one
    def test_native(self):
        res = self.run_configure(["-DCPU_BASELINE=NATIVE"])
        self.check_features(res, MatchResult(["SSE SSE2 SSE3 SSSE3 SSE4_1 POPCNT SSE4_2 AVX FP16 AVX2 FMA3 AVX_512F AVX512_COMMON AVX512_SKX AVX512_CNL AVX512_CLX AVX512_ICL", "NATIVE", None, "SSE4_1 SSE4_2 AVX FP16 AVX2 AVX512_SKX"]))


class Test_AArch64(TestBase):
    def run_configure(self, args):
        return subprocess.run(["cmake", "-GNinja", "-DCMAKE_TOOLCHAIN_FILE={}/platforms/linux/aarch64-gnu.toolchain.cmake".format(OPENCV)] + args + [OPENCV], capture_output=True, cwd=BUILD)

    def test_default(self):
        res = self.run_configure([])
        self.check_features(res, MatchResult(["NEON FP16", "DETECT", None, None]))


if __name__ == '__main__':
    print("OPENCV={}".format(OPENCV))
    print("BUILD={}".format(BUILD))
    os.makedirs(BUILD, exist_ok=True)
    unittest.main(verbosity=3)