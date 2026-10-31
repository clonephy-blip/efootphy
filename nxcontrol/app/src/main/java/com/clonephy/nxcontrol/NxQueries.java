package com.clonephy.nxcontrol;

public final class NxQueries {
    private NxQueries() {}
    public static String telemetry() {
        return "CPU=$(for z in /sys/class/thermal/thermal_zone*; do t=$(cat $z/type 2>/dev/null); case $t in cpuss*|apc*) cat $z/temp 2>/dev/null;; esac; done|sort -nr|head -1);GPU=$(for z in /sys/class/thermal/thermal_zone*; do t=$(cat $z/type 2>/dev/null); case $t in gpuss*) cat $z/temp 2>/dev/null;; esac; done|sort -nr|head -1);T=$(awk '/MemTotal/{print $2}' /proc/meminfo);A=$(awk '/MemAvailable/{print $2}' /proc/meminfo);B=$(cat /sys/class/power_supply/battery/charge_control_limit 2>/dev/null);echo $CPU'|'$GPU'|'$T'|'$A'|'$B";
    }
    public static String bypassState() { return "cat /sys/class/power_supply/battery/charge_control_limit 2>/dev/null"; }
    public static String displayInfo() { return "wm size; wm density; echo minHz=$(settings get system min_refresh_rate); echo peakHz=$(settings get system peak_refresh_rate)"; }
    public static String saveScript(String safeName, String base64) {
        return "mkdir -p /sdcard/NXControl/scripts; echo '"+base64+"' | base64 -d > /sdcard/NXControl/scripts/"+safeName+"; chmod 700 /sdcard/NXControl/scripts/"+safeName+"; echo SAVED";
    }
    public static String runScript(String safeName) { return "sh /sdcard/NXControl/scripts/"+safeName; }
    public static String listScripts() { return "mkdir -p /sdcard/NXControl/scripts; ls -1 /sdcard/NXControl/scripts 2>/dev/null"; }
}
