package com.clonephy.nxcontrol;

public final class NxActions {
    public static final String BYPASS = "/sys/class/power_supply/battery/charge_control_limit";
    private NxActions() {}

    public static String bypassOn() {
        return "chmod 644 " + BYPASS + "; echo 4 > " + BYPASS + "; cat " + BYPASS;
    }
    public static String bypassOff() {
        return "chmod 644 " + BYPASS + "; echo 0 > " + BYPASS + "; chmod 444 " + BYPASS + "; cat " + BYPASS;
    }
    public static String clean() {
        return "B=$(awk '/MemAvailable/{print $2}' /proc/meminfo); am kill-all; sleep 1; A=$(awk '/MemAvailable/{print $2}' /proc/meminfo); echo $B $A";
    }
    public static String boost30() {
        return "(G0=$(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor);G4=$(cat /sys/devices/system/cpu/cpufreq/policy4/scaling_governor);G7=$(cat /sys/devices/system/cpu/cpufreq/policy7/scaling_governor);GMN=$(cat /sys/class/kgsl/kgsl-3d0/min_clock_mhz);R1=$(settings get system min_refresh_rate);R2=$(settings get system peak_refresh_rate);echo performance >/sys/devices/system/cpu/cpufreq/policy0/scaling_governor;echo performance >/sys/devices/system/cpu/cpufreq/policy4/scaling_governor;echo performance >/sys/devices/system/cpu/cpufreq/policy7/scaling_governor;echo 400 >/sys/class/kgsl/kgsl-3d0/min_clock_mhz;settings put system min_refresh_rate 144.0;settings put system peak_refresh_rate 144.0;sleep 30;echo $G0 >/sys/devices/system/cpu/cpufreq/policy0/scaling_governor;echo $G4 >/sys/devices/system/cpu/cpufreq/policy4/scaling_governor;echo $G7 >/sys/devices/system/cpu/cpufreq/policy7/scaling_governor;echo $GMN >/sys/class/kgsl/kgsl-3d0/min_clock_mhz;settings put system min_refresh_rate $R1;settings put system peak_refresh_rate $R2) >/dev/null 2>&1 & echo BOOST_30S";
    }
    public static String cool60() {
        return "(G0=$(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor);G4=$(cat /sys/devices/system/cpu/cpufreq/policy4/scaling_governor);G7=$(cat /sys/devices/system/cpu/cpufreq/policy7/scaling_governor);GMX=$(cat /sys/class/kgsl/kgsl-3d0/max_clock_mhz);R1=$(settings get system min_refresh_rate);R2=$(settings get system peak_refresh_rate);echo powersave >/sys/devices/system/cpu/cpufreq/policy0/scaling_governor;echo powersave >/sys/devices/system/cpu/cpufreq/policy4/scaling_governor;echo powersave >/sys/devices/system/cpu/cpufreq/policy7/scaling_governor;echo 400 >/sys/class/kgsl/kgsl-3d0/max_clock_mhz;settings put system min_refresh_rate 60.0;settings put system peak_refresh_rate 60.0;sleep 60;echo $G0 >/sys/devices/system/cpu/cpufreq/policy0/scaling_governor;echo $G4 >/sys/devices/system/cpu/cpufreq/policy4/scaling_governor;echo $G7 >/sys/devices/system/cpu/cpufreq/policy7/scaling_governor;echo $GMX >/sys/class/kgsl/kgsl-3d0/max_clock_mhz;settings put system min_refresh_rate $R1;settings put system peak_refresh_rate $R2) >/dev/null 2>&1 & echo COOL_60S";
    }
    public static String panicReset() {
        return "wm size reset; wm density reset; settings put system min_refresh_rate 120.0; settings put system peak_refresh_rate 120.0; echo schedutil >/sys/devices/system/cpu/cpufreq/policy0/scaling_governor; echo schedutil >/sys/devices/system/cpu/cpufreq/policy4/scaling_governor; echo schedutil >/sys/devices/system/cpu/cpufreq/policy7/scaling_governor; echo 305 >/sys/class/kgsl/kgsl-3d0/min_clock_mhz; echo 587 >/sys/class/kgsl/kgsl-3d0/max_clock_mhz; chmod 644 " + BYPASS + "; echo 0 > " + BYPASS + "; chmod 444 " + BYPASS + "; echo RESET_OK";
    }
}
