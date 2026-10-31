package com.clonephy.nxcontrol;

import android.app.AlertDialog;
import android.content.Context;
import android.content.SharedPreferences;
import android.graphics.Color;
import android.widget.EditText;
import android.widget.LinearLayout;

public final class ProfileDialogs {
    private ProfileDialogs() {}
    private static EditText field(Context c, String hint, String value) {
        EditText e = new EditText(c);
        e.setHint(hint);
        e.setText(value);
        e.setTextColor(Color.WHITE);
        e.setHintTextColor(Color.GRAY);
        e.setSingleLine(true);
        return e;
    }
    public static void edit(Context c, SharedPreferences prefs, String pkg) {
        LinearLayout box = new LinearLayout(c);
        box.setOrientation(LinearLayout.VERTICAL);
        int p = (int)(18*c.getResources().getDisplayMetrics().density);
        box.setPadding(p,0,p,0);
        EditText w = field(c,"Width, ví dụ 1080",prefs.getString("p_"+pkg+"_w",""));
        EditText h = field(c,"Height, ví dụ 2340",prefs.getString("p_"+pkg+"_h",""));
        EditText dpi = field(c,"DPI, ví dụ 420",prefs.getString("p_"+pkg+"_d",""));
        EditText hz = field(c,"Hz: 60 / 90 / 120 / 144",prefs.getString("p_"+pkg+"_hz","120"));
        box.addView(w); box.addView(h); box.addView(dpi); box.addView(hz);
        new AlertDialog.Builder(c)
            .setTitle("Profile: "+pkg)
            .setMessage("Resolution/DPI sẽ chỉ áp khi app foreground và tự reset khi thoát.")
            .setView(box)
            .setPositiveButton("Lưu", (d,x) -> prefs.edit()
                .putString("p_"+pkg+"_w",w.getText().toString().trim())
                .putString("p_"+pkg+"_h",h.getText().toString().trim())
                .putString("p_"+pkg+"_d",dpi.getText().toString().trim())
                .putString("p_"+pkg+"_hz",hz.getText().toString().trim()).apply())
            .setNeutralButton("Xóa profile", (d,x) -> prefs.edit()
                .remove("p_"+pkg+"_w").remove("p_"+pkg+"_h").remove("p_"+pkg+"_d").remove("p_"+pkg+"_hz").apply())
            .setNegativeButton("Hủy",null).show();
    }
}
