import unittest
import subprocess
import os
import re
import logging
from pathlib import Path
import shutil

log = logging.getLogger(__name__)
OPENCV = Path(os.environ["OPENCV"])
BUILD = Path(os.environ["BUILD"])

class MatchResult:
    def __init__(self, arr):
        self.baseline = arr[0]
        self.requested = arr[1]
        self.disabled = arr[2]
        self.dispatched = arr[3]
    def __repr__(self):
        return "({}, {}, {}, {})".format(self.baseline, self.requested, self.disabled, self.dispatched)

# CPU_BASELINE=DETECT
# CPU_BASELINE_DETECT=ON
# CPU_BASELINE_DISABLE=;VFPV3
# CPU_BASELINE_FINAL=NEON;FP16
# CPU_BASELINE_FLAGS=
# CPU_BASELINE_REQUIRE=;NEON
# CPU_DISPATCH=NEON_FP16;NEON_BF16;NEON_DOTPROD
def parse_vars():
    def get_group(match, idx=1):
        return match.group(idx) if match is not None else ""
    with open(BUILD / "CMakeVars.txt", "r") as f:
        contents = f.read()
        return MatchResult([
            get_group(re.search(r"^CPU_BASELINE_FINAL=(.*)$", contents, re.MULTILINE)),
            get_group(re.search(r"^CPU_BASELINE=(.*)$", contents, re.MULTILINE)),
            get_group(re.search(r"^CPU_BASELINE_DISABLE=(.*)$", contents, re.MULTILINE)),
            get_group(re.search(r"^CPU_DISPATCH=(.*)$", contents, re.MULTILINE)),
        ])


class TestBase(unittest.TestCase):
    def check_features(self, args, expected):
        log.info("process arguments: %s", str(args))
        popen_res = self.run_configure(args)
        log.info("process stdout:\n%s", popen_res.stdout.decode("utf-8"))
        log.info("process stderr:\n%s", popen_res.stderr.decode("utf-8"))
        self.assertEqual(popen_res.returncode, 0)
        actual = parse_vars()
        self.assertDictEqual(actual.__dict__, MatchResult(expected).__dict__)

    def setUp(self):
        log.info("\n===========\n%s\n===========\n", self.id())
        if BUILD.exists():
            log.info("cleanup: %s", BUILD)
            shutil.rmtree(BUILD)
        os.makedirs(BUILD, exist_ok=True)


class Test_x86_64(TestBase):
    def run_configure(self, args):
        return subprocess.run(["cmake", "-GNinja"] + args + [OPENCV], capture_output=True, cwd=BUILD)

    def test_default(self):
        self.check_features(
            [],
            ["SSE;SSE2;SSE3", "SSE3", "", "SSE4_1;SSE4_2;AVX;FP16;AVX2;AVX512_SKX"])

    def test_disable_sse2(self):
        self.check_features(
            ["-DCPU_BASELINE_DISABLE=SSE2"],
            ["SSE", "SSE3", "SSE2", "SSE4_1;SSE4_2;AVX;FP16;AVX2;AVX512_SKX"])

    def test_detect(self):
        self.check_features(
            ["-DCPU_BASELINE=DETECT"],
            ["SSE;SSE2", "DETECT", "", "SSE4_1;SSE4_2;AVX;FP16;AVX2;AVX512_SKX"])

    # will fail on platforms other than specific one
    def test_native(self):
        self.check_features(
            ["-DCPU_BASELINE=NATIVE"],
            ["SSE;SSE2;SSE3;SSSE3;SSE4_1;POPCNT;SSE4_2;AVX;FP16;AVX2;FMA3;AVX_512F;AVX512_COMMON;AVX512_SKX;AVX512_CNL;AVX512_CLX;AVX512_ICL", "NATIVE", "", "SSE4_1;SSE4_2;AVX;FP16;AVX2;AVX512_SKX"])


class Test_AArch64(TestBase):
    def run_configure(self, args):
        return subprocess.run(["cmake", "-GNinja", "-DCMAKE_TOOLCHAIN_FILE={}/platforms/linux/aarch64-gnu.toolchain.cmake".format(OPENCV)] + args + [OPENCV], capture_output=True, cwd=BUILD)

    def test_default(self):
        self.check_features(
            [],
            ["NEON;FP16", "NEON;FP16", ";VFPV3", "NEON_FP16;NEON_BF16;NEON_DOTPROD"])

    def test_detect(self):
        self.check_features(
            ["-DCPU_BASELINE=DETECT"],
            ["NEON;FP16", "DETECT", ";VFPV3", "NEON_FP16;NEON_BF16;NEON_DOTPROD"])

    def test_disable_fp16(self):
        self.check_features(
            ["-DCPU_BASELINE_DISABLE=FP16"],
            ["NEON", "NEON;FP16", "FP16;VFPV3", "NEON_FP16;NEON_BF16;NEON_DOTPROD"])
        self.check_features(
            ["-DCPU_BASELINE=NEON", "-DCPU_BASELINE_DISABLE=FP16"],
            ["NEON", "NEON", "FP16;VFPV3", "NEON_FP16;NEON_BF16;NEON_DOTPROD"])

    def test_disable_neon(self):
        self.check_features(
            ["-DCPU_BASELINE_DISABLE=NEON", "-DENABLE_NEON=OFF"],
            ["", "NEON;FP16", "NEON;VFPV3", "NEON_FP16;NEON_BF16;NEON_DOTPROD"])


class Test_ARM(TestBase):
    def run_configure(self, args):
        return subprocess.run(["cmake", "-GNinja", "-DCMAKE_TOOLCHAIN_FILE={}/platforms/linux/arm-gnueabi.toolchain.cmake".format(OPENCV)] + args + [OPENCV], capture_output=True, cwd=BUILD)

    def test_default(self):
        self.check_features(
            [],
            ["", "DETECT", ";VFPV3;NEON", ""])

    def test_neon(self):
        self.check_features(
            ["-DCPU_BASELINE=NEON"],
            ["", "NEON", ";VFPV3;NEON", ""])

    def test_fp16(self):
        self.check_features(
            ["-DCPU_BASELINE=FP16"],
            ["", "FP16", ";VFPV3;NEON", ""])


class Test_RISCV(TestBase):
    def run_configure(self, args):
        return subprocess.run(["cmake", "-GNinja", "-DCMAKE_TOOLCHAIN_FILE={}/platforms/linux/riscv64-gcc.toolchain.cmake".format(OPENCV), "-DGNU_MACHINE=riscv64-linux-gnu"] + args + [OPENCV], capture_output=True, cwd=BUILD)

    def test_default(self):
        self.check_features(
            [],
            ["", "DETECT", "", ""])


if __name__ == '__main__':
    logging.basicConfig(filename='test_log.txt', filemode='w', level=logging.INFO)
    print("OPENCV={}".format(OPENCV))
    print("BUILD={}".format(BUILD))
    assert(OPENCV.exists())
    unittest.main(verbosity=3)