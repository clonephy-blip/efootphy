package com.clonephy.nxcontrol;

import com.topjohnwu.superuser.Shell;
import java.util.List;

public final class RootBridge {
    static {
        Shell.enableVerboseLogging = false;
        Shell.setDefaultBuilder(Shell.Builder.create().setFlags(Shell.FLAG_MOUNT_MASTER).setTimeout(10));
    }
    private RootBridge() {}
    public static String run(String command) {
        Shell.Result result = Shell.cmd(command).exec();
        List<String> out = result.getOut();
        return String.join("\n", out);
    }
    public static boolean ok(String command) {
        return Shell.cmd(command).exec().isSuccess();
    }
}
