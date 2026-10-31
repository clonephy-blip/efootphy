package com.clonephy.nxcontrol;

import android.accessibilityservice.AccessibilityService;
import android.content.Intent;
import android.graphics.Color;
import android.graphics.PixelFormat;
import android.graphics.drawable.GradientDrawable;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.View;
import android.view.WindowManager;
import android.view.accessibility.AccessibilityEvent;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import java.util.Collections;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class GameOverlayAccessibilityService extends AccessibilityService {
    private WindowManager wm;
    private TextView bubble;
    private LinearLayout menu;
    private WindowManager.LayoutParams bubbleLp;
    private final ExecutorService io = Executors.newSingleThreadExecutor();
    private String activePackage = "";

    @Override public void onServiceConnected() {
        wm = (WindowManager) getSystemService(WINDOW_SERVICE);
    }

    @Override public void onAccessibilityEvent(AccessibilityEvent event) {
        if (event == null || event.getPackageName() == null) return;
        String pkg = event.getPackageName().toString();
        if (pkg.equals(getPackageName())) return;
        boolean allowed = selectedApps().contains(pkg);
        if (allowed) {
            activePackage = pkg;
            showBubble();
        } else if (!pkg.equals(activePackage)) {
            activePackage = "";
            hideAll();
        }
    }

    @Override public void onInterrupt() { }

    private Set<String> selectedApps() {
        return new HashSet<>(getSharedPreferences("nx", MODE_PRIVATE)
                .getStringSet("apps", Collections.emptySet()));
    }

    private int dp(int n) {
        return (int) (n * getResources().getDisplayMetrics().density + 0.5f);
    }

    private GradientDrawable bg(int color, int radius) {
        GradientDrawable d = new GradientDrawable();
        d.setColor(color);
        d.setCornerRadius(dp(radius));
        return d;
    }

    private WindowManager.LayoutParams lp(int w, int h, int x, int y) {
        WindowManager.LayoutParams p = new WindowManager.LayoutParams(
                w, h,
                WindowManager.LayoutParams.TYPE_ACCESSIBILITY_OVERLAY,
                WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE |
                        WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS,
                PixelFormat.TRANSLUCENT);
        p.gravity = Gravity.TOP | Gravity.END;
        p.x = x;
        p.y = y;
        return p;
    }

    private void showBubble() {
        if (wm == null || bubble != null) return;
        bubble = new TextView(this);
        bubble.setText("NX");
        bubble.setTextColor(Color.WHITE);
        bubble.setTextSize(11);
        bubble.setTypeface(null, 1);
        bubble.setGravity(Gravity.CENTER);
        bubble.setBackground(bg(Color.rgb(139, 84, 39), 24));
        bubbleLp = lp(dp(42), dp(42), dp(8), dp(170));
        bubble.setOnTouchListener(new View.OnTouchListener() {
            float downX, downY;
            int startX, startY;
            boolean moved;
            @Override public boolean onTouch(View v, MotionEvent e) {
                if (e.getAction() == MotionEvent.ACTION_DOWN) {
                    downX = e.getRawX(); downY = e.getRawY();
                    startX = bubbleLp.x; startY = bubbleLp.y; moved = false;
                    return true;
                }
                if (e.getAction() == MotionEvent.ACTION_MOVE) {
                    float dx = e.getRawX() - downX;
                    float dy = e.getRawY() - downY;
                    if (Math.abs(dx) > dp(4) || Math.abs(dy) > dp(4)) moved = true;
                    bubbleLp.x = Math.max(0, startX - (int) dx);
                    bubbleLp.y = Math.max(0, startY + (int) dy);
                    try { wm.updateViewLayout(bubble, bubbleLp); } catch (Throwable ignored) { }
                    return true;
                }
                if (e.getAction() == MotionEvent.ACTION_UP) {
                    if (!moved) toggleMenu();
                    return true;
                }
                return false;
            }
        });
        wm.addView(bubble, bubbleLp);
    }

    private Button button(String text) {
        Button b = new Button(this);
        b.setText(text);
        b.setAllCaps(false);
        b.setTextColor(Color.rgb(249, 226, 207));
        b.setTextSize(12);
        b.setBackground(bg(Color.rgb(58, 35, 21), 12));
        return b;
    }

    private void toggleMenu() {
        if (menu != null) {
            try { wm.removeView(menu); } catch (Throwable ignored) { }
            menu = null;
            return;
        }
        menu = new LinearLayout(this);
        menu.setOrientation(LinearLayout.VERTICAL);
        menu.setPadding(dp(6), dp(6), dp(6), dp(6));
        menu.setBackground(bg(Color.rgb(30, 20, 14), 16));
        Button bypass = button("🔌 Bypass");
        Button boost = button("⚡ Boost 30s");
        Button cool = button("❄ Cool 60s");
        Button clean = button("🧹 Clean");
        Button open = button("⚙ Control Center");
        for (Button b : new Button[]{bypass, boost, cool, clean, open}) {
            LinearLayout.LayoutParams p = new LinearLayout.LayoutParams(dp(146), dp(44));
            p.setMargins(0, dp(2), 0, dp(2));
            menu.addView(b, p);
        }
        wm.addView(menu, lp(dp(158), -2, dp(56), Math.max(dp(70), bubbleLp.y - dp(75))));
        bypass.setOnClickListener(v -> toggleBypass());
        boost.setOnClickListener(v -> rootAction(NxActions.boost30(), "Boost 30s"));
        cool.setOnClickListener(v -> rootAction(NxActions.cool60(), "Cool 60s"));
        clean.setOnClickListener(v -> rootAction(NxActions.clean(), "Đã dọn app nền"));
        open.setOnClickListener(v -> {
            Intent i = new Intent(this, MainActivity.class);
            i.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
            startActivity(i);
            closeMenu();
        });
    }

    private void toggleBypass() {
        io.execute(() -> {
            String state = RootBridge.run(NxQueries.bypassState());
            String result = RootBridge.run("4".equals(state.trim()) ? NxActions.bypassOff() : NxActions.bypassOn());
            runOnUiThreadToast("Bypass: " + ("4".equals(result.trim()) ? "ON" : "OFF"));
        });
        closeMenu();
    }

    private void rootAction(String command, String label) {
        io.execute(() -> {
            RootBridge.run(command);
            runOnUiThreadToast(label);
        });
        closeMenu();
    }

    private void runOnUiThreadToast(String text) {
        new android.os.Handler(getMainLooper()).post(() ->
                Toast.makeText(this, text, Toast.LENGTH_SHORT).show());
    }

    private void closeMenu() {
        if (menu != null) {
            try { wm.removeView(menu); } catch (Throwable ignored) { }
            menu = null;
        }
    }

    private void hideAll() {
        closeMenu();
        if (bubble != null) {
            try { wm.removeView(bubble); } catch (Throwable ignored) { }
            bubble = null;
        }
    }

    @Override public void onDestroy() {
        hideAll();
        io.shutdownNow();
        super.onDestroy();
    }
}
