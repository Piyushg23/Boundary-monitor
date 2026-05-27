"""
launcher.py — Dark-themed Tkinter launcher window.
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

from utils.config import CFG, GROQ_API_KEY, ENV_PATH


def launch_gui():
    from pipeline import run

    selected_source = {"value": None, "demo": True}

    root = tk.Tk()
    root.title("Boundary Monitor v3 -- Launcher")
    root.configure(bg="#0d0d0d")
    root.resizable(False, False)

    BG     = "#0d0d0d"
    PANEL  = "#141414"
    BORDER = "#1f3d1f"
    GREEN  = "#00ff64"
    DIM    = "#3a6e40"
    WHITE  = "#d0dcd0"
    GOLD   = "#d4b04a"
    RED    = "#ff4444"
    FONT_H = ("Courier New", 13, "bold")
    FONT_B = ("Courier New", 10, "bold")
    FONT_S = ("Courier New",  9)

    # ── Header ────────────────────────────────────────────────────────────────
    header = tk.Frame(root, bg=BG, pady=8)
    header.pack(fill="x", padx=16)
    tk.Label(header, text="[*] BOUNDARY MONITOR v3.0",
             font=("Courier New", 15, "bold"), fg=GREEN, bg=BG).pack()
    tk.Label(header, text="IG Defence  /  Constems-AI Grade  /  Groq LLM",
             font=FONT_S, fg=DIM, bg=BG).pack()

    tk.Frame(root, height=1, bg=BORDER).pack(fill="x", padx=16, pady=4)

    # ── Groq API Key panel ────────────────────────────────────────────────────
    _placeholder = {"", "your_groq_api_key_here", "YOUR_GROQ_API_KEY_HERE"}
    key_ok = GROQ_API_KEY not in _placeholder

    key_panel = tk.LabelFrame(
        root, text=" GROQ API KEY ", font=FONT_B, fg=GREEN, bg=PANEL,
        bd=1, relief="solid", highlightbackground=BORDER, labelanchor="nw",
    )
    key_panel.pack(fill="x", padx=16, pady=6, ipady=6)

    key_row = tk.Frame(key_panel, bg=PANEL)
    key_row.pack(fill="x", padx=10, pady=4)

    key_var = tk.StringVar(value=GROQ_API_KEY if key_ok else "")
    key_entry = tk.Entry(
        key_row, textvariable=key_var, width=48, show="*",
        bg="#0a1a0a", fg=GREEN, insertbackground=GREEN,
        relief="flat", font=FONT_S,
    )
    key_entry.pack(side="left", padx=(0, 8))

    def toggle_show():
        key_entry.config(show="" if key_entry.cget("show") == "*" else "*")

    tk.Button(
        key_row, text="Show", command=toggle_show,
        bg="#0a2a0a", fg=GREEN, activebackground="#0f3a0f", activeforeground=GREEN,
        relief="flat", font=FONT_S, padx=6,
    ).pack(side="left", padx=(0, 8))

    def save_key():
        k = key_var.get().strip()
        if not k:
            messagebox.showerror("Empty key", "Please enter your Groq API key.")
            return
        try:
            env_file = ENV_PATH
            lines = []
            replaced = False
            if env_file.exists():
                for line in env_file.read_text().splitlines():
                    if line.startswith("GROQ_API_KEY"):
                        lines.append(f"GROQ_API_KEY={k}")
                        replaced = True
                    else:
                        lines.append(line)
            if not replaced:
                lines.append(f"GROQ_API_KEY={k}")
            env_file.write_text("\n".join(lines) + "\n")
            # Also set in current process environment immediately
            os.environ["GROQ_API_KEY"] = k
            key_status_var.set("  Key saved to .env and active for this session")
            key_status_lbl.config(fg=GREEN)
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    tk.Button(
        key_row, text="Save to .env", command=save_key,
        bg="#0a2a0a", fg=GREEN, activebackground="#0f3a0f", activeforeground=GREEN,
        relief="flat", font=FONT_B, padx=8,
        highlightthickness=1, highlightbackground=BORDER,
    ).pack(side="left")

    key_status_var = tk.StringVar()
    if key_ok:
        masked = GROQ_API_KEY[:8] + "..." + GROQ_API_KEY[-4:]
        key_status_var.set(f"  Key loaded: {masked}")
        status_fg = GREEN
    else:
        key_status_var.set(f"  No key found at: {ENV_PATH}  -- enter key above and click Save")
        status_fg = RED

    key_status_lbl = tk.Label(key_panel, textvariable=key_status_var,
                               font=FONT_S, fg=status_fg, bg=PANEL, anchor="w")
    key_status_lbl.pack(fill="x", padx=10, pady=(0, 4))

    tk.Label(key_panel,
             text="  Get a free key at: https://console.groq.com",
             font=FONT_S, fg=DIM, bg=PANEL, anchor="w").pack(fill="x", padx=10, pady=(0, 4))

    tk.Frame(root, height=1, bg=BORDER).pack(fill="x", padx=16, pady=4)

    # ── Source panel ──────────────────────────────────────────────────────────
    src_panel = tk.LabelFrame(
        root, text=" VIDEO SOURCE ", font=FONT_B, fg=GREEN, bg=PANEL,
        bd=1, relief="solid", highlightbackground=BORDER, labelanchor="nw",
    )
    src_panel.pack(fill="x", padx=16, pady=6, ipady=6)

    mode_var = tk.StringVar(value="demo")

    def update_mode(*_):
        m = mode_var.get()
        if m == "demo":
            file_entry.config(state="disabled")
            browse_btn.config(state="disabled")
            cam_spin.config(state="disabled")
            selected_source["demo"] = True
            selected_source["value"] = None
        elif m == "file":
            file_entry.config(state="normal")
            browse_btn.config(state="normal")
            cam_spin.config(state="disabled")
            selected_source["demo"] = False
        elif m == "webcam":
            file_entry.config(state="disabled")
            browse_btn.config(state="disabled")
            cam_spin.config(state="normal")
            selected_source["demo"] = False
            selected_source["value"] = int(cam_idx.get())

    def _radio(parent, text, value, fg=None):
        fg = fg or WHITE
        return tk.Radiobutton(
            parent, text=text, variable=mode_var, value=value, command=update_mode,
            fg=fg, bg=PANEL, selectcolor="#002200",
            activebackground=PANEL, activeforeground=fg, font=FONT_B,
        )

    demo_row = tk.Frame(src_panel, bg=PANEL)
    demo_row.pack(fill="x", padx=10, pady=3)
    _radio(demo_row, "Built-in Demo  (synthetic airfield scene)", "demo", GREEN).pack(side="left")

    file_row = tk.Frame(src_panel, bg=PANEL)
    file_row.pack(fill="x", padx=10, pady=3)
    _radio(file_row, "Upload Video File", "file").pack(side="left")
    file_path_var = tk.StringVar(value="No file selected")
    file_entry = tk.Entry(
        file_row, textvariable=file_path_var, width=34, state="disabled",
        bg="#0a1a0a", fg=WHITE, insertbackground=GREEN,
        disabledbackground="#080808", disabledforeground="#444",
        relief="flat", font=FONT_S,
    )
    file_entry.pack(side="left", padx=(8, 4))

    def browse_file():
        path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.m4v *.ts *.flv"),
                ("All files",   "*.*"),
            ],
        )
        if path:
            file_path_var.set(path)
            selected_source["value"] = path
            selected_source["demo"] = False

    browse_btn = tk.Button(
        file_row, text="Browse...", command=browse_file, state="disabled",
        bg="#0a2a0a", fg=GREEN, activebackground="#0f3a0f", activeforeground=GREEN,
        relief="flat", font=FONT_B, padx=8, pady=2,
        highlightthickness=1, highlightbackground=BORDER,
    )
    browse_btn.pack(side="left")

    cam_row = tk.Frame(src_panel, bg=PANEL)
    cam_row.pack(fill="x", padx=10, pady=3)
    _radio(cam_row, "Webcam / Camera index:", "webcam").pack(side="left")
    cam_idx = tk.StringVar(value="0")
    cam_spin = tk.Spinbox(
        cam_row, from_=0, to=9, textvariable=cam_idx, width=3, state="disabled",
        bg="#0a1a0a", fg=GREEN, buttonbackground="#0a2a0a",
        disabledbackground="#080808", disabledforeground="#444",
        relief="flat", font=FONT_B,
    )
    cam_spin.pack(side="left", padx=6)

    tk.Frame(root, height=1, bg=BORDER).pack(fill="x", padx=16, pady=4)

    # ── Options ───────────────────────────────────────────────────────────────
    opt_panel = tk.LabelFrame(
        root, text=" OPTIONS ", font=FONT_B, fg=GREEN, bg=PANEL,
        bd=1, relief="solid", highlightbackground=BORDER, labelanchor="nw",
    )
    opt_panel.pack(fill="x", padx=16, pady=6, ipady=4)
    opt_inner = tk.Frame(opt_panel, bg=PANEL)
    opt_inner.pack(fill="x", padx=10)

    night_var     = tk.BooleanVar(value=False)
    denoise_var   = tk.BooleanVar(value=False)
    stabilise_var = tk.BooleanVar(value=False)
    save_var      = tk.BooleanVar(value=False)

    for text, var, col in [
        ("Night Mode",      night_var,     GOLD),
        ("Denoise",         denoise_var,   WHITE),
        ("Stabilise",       stabilise_var, WHITE),
        ("Save Output MP4", save_var,      GREEN),
    ]:
        tk.Checkbutton(
            opt_inner, text=text, variable=var,
            fg=col, bg=PANEL, selectcolor="#001a00",
            activebackground=PANEL, activeforeground=col, font=FONT_S,
        ).pack(side="left", padx=8, pady=2)

    tk.Frame(root, height=1, bg=BORDER).pack(fill="x", padx=16, pady=4)

    # ── Launch ────────────────────────────────────────────────────────────────
    status_var = tk.StringVar(value="Ready -- select source and press LAUNCH")
    tk.Label(root, textvariable=status_var, font=FONT_S, fg=DIM, bg=BG, anchor="w"
             ).pack(fill="x", padx=18, pady=(0, 4))

    def do_launch():
        mode = mode_var.get()
        if mode == "file":
            path = file_path_var.get()
            if not path or path == "No file selected":
                messagebox.showerror("No file", "Please select a video file first.")
                return
            if not os.path.exists(path):
                messagebox.showerror("File not found", f"Cannot find:\n{path}")
                return
            selected_source["value"] = path
            selected_source["demo"] = False
        elif mode == "webcam":
            try:
                selected_source["value"] = int(cam_idx.get())
            except ValueError:
                messagebox.showerror("Invalid index", "Camera index must be an integer.")
                return
            selected_source["demo"] = False

        # Re-read key from entry in case user typed it but didn't save
        live_key = key_var.get().strip()
        if live_key and live_key not in _placeholder:
            os.environ["GROQ_API_KEY"] = live_key

        CFG["night_mode"] = night_var.get()
        CFG["denoise"]    = denoise_var.get()
        CFG["stabilise"]  = stabilise_var.get()

        status_var.set("Launching pipeline...")
        root.update()
        root.withdraw()

        try:
            run(
                source=selected_source["value"],
                demo_mode=selected_source["demo"],
                save_output=save_var.get(),
            )
        except Exception as e:
            messagebox.showerror("Pipeline error", str(e))
        finally:
            root.destroy()

    tk.Button(
        root, text=">>  LAUNCH MONITOR", command=do_launch,
        bg="#003300", fg=GREEN, activebackground="#004d00", activeforeground=GREEN,
        font=("Courier New", 12, "bold"), relief="flat", pady=10,
        highlightthickness=1, highlightbackground=GREEN,
    ).pack(fill="x", padx=16, pady=(4, 12))

    root.bind("<Return>", lambda e: do_launch())
    root.mainloop()
