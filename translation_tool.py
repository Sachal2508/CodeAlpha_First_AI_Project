
import tkinter as tk
from tkinter import ttk, messagebox
from deep_translator import GoogleTranslator

# --- List of languages supported ---
LANGUAGES = {
    "English": "en",
    "Urdu": "ur",
    "French": "fr",
    "Spanish": "es",
    "German": "de",
    "Arabic": "ar",
    "Chinese (Simplified)": "zh-CN",
    "Japanese": "ja",
    "Hindi": "hi",
    "Turkish": "tr",
    "Russian": "ru",
    "Portuguese": "pt",
    "Italian": "it",
    "Korean": "ko",
}

# --- Main Translation Function ---
def translate_text():
    input_text = input_box.get("1.0", tk.END).strip()  # Get text from input box

    if not input_text:
        messagebox.showwarning("Empty Input", "Please enter some text to translate.")
        return

    source_lang = LANGUAGES[source_var.get()]  # Get selected source language code
    target_lang = LANGUAGES[target_var.get()]  # Get selected target language code

    try:
        # Translate using GoogleTranslator (free, no API key)
        translated = GoogleTranslator(source=source_lang, target=target_lang).translate(input_text)

        # Show result in output box
        output_box.config(state="normal")
        output_box.delete("1.0", tk.END)
        output_box.insert(tk.END, translated)
        output_box.config(state="disabled")

    except Exception as e:
        messagebox.showerror("Translation Error", str(e))

# --- Copy Output to Clipboard ---
def copy_text():
    result = output_box.get("1.0", tk.END).strip()
    if result:
        root.clipboard_clear()
        root.clipboard_append(result)
        messagebox.showinfo("Copied", "Translated text copied to clipboard!")

# --- Clear All ---
def clear_all():
    input_box.delete("1.0", tk.END)
    output_box.config(state="normal")
    output_box.delete("1.0", tk.END)
    output_box.config(state="disabled")

# ============================================================
# BUILD THE UI
# ============================================================
root = tk.Tk()
root.title("Language Translator")
root.geometry("650x500")
root.resizable(False, False)
root.configure(bg="#1e1e2e")

# --- Title ---
tk.Label(root, text="🌐 Language Translator", font=("Arial", 18, "bold"),
         bg="#1e1e2e", fg="#cdd6f4").pack(pady=10)

# --- Language Selection Row ---
lang_frame = tk.Frame(root, bg="#1e1e2e")
lang_frame.pack(pady=5)

tk.Label(lang_frame, text="From:", font=("Arial", 11), bg="#1e1e2e", fg="#a6e3a1").grid(row=0, column=0, padx=5)
source_var = tk.StringVar(value="English")
source_menu = ttk.Combobox(lang_frame, textvariable=source_var,
                            values=list(LANGUAGES.keys()), width=20, state="readonly")
source_menu.grid(row=0, column=1, padx=5)

tk.Label(lang_frame, text="To:", font=("Arial", 11), bg="#1e1e2e", fg="#f38ba8").grid(row=0, column=2, padx=5)
target_var = tk.StringVar(value="Urdu")
target_menu = ttk.Combobox(lang_frame, textvariable=target_var,
                             values=list(LANGUAGES.keys()), width=20, state="readonly")
target_menu.grid(row=0, column=3, padx=5)

# --- Input Box ---
tk.Label(root, text="Enter Text:", font=("Arial", 11), bg="#1e1e2e", fg="#cdd6f4").pack(anchor="w", padx=20)
input_box = tk.Text(root, height=6, width=70, font=("Arial", 11),
                    bg="#313244", fg="#cdd6f4", insertbackground="white", relief="flat")
input_box.pack(padx=20, pady=5)

# --- Buttons Row ---
btn_frame = tk.Frame(root, bg="#1e1e2e")
btn_frame.pack(pady=5)

tk.Button(btn_frame, text="Translate", command=translate_text,
          bg="#89b4fa", fg="#1e1e2e", font=("Arial", 11, "bold"),
          padx=20, pady=5, relief="flat").grid(row=0, column=0, padx=10)

tk.Button(btn_frame, text="Copy Result", command=copy_text,
          bg="#a6e3a1", fg="#1e1e2e", font=("Arial", 11),
          padx=20, pady=5, relief="flat").grid(row=0, column=1, padx=10)

tk.Button(btn_frame, text="Clear", command=clear_all,
          bg="#f38ba8", fg="#1e1e2e", font=("Arial", 11),
          padx=20, pady=5, relief="flat").grid(row=0, column=2, padx=10)

# --- Output Box ---
tk.Label(root, text="Translated Text:", font=("Arial", 11), bg="#1e1e2e", fg="#cdd6f4").pack(anchor="w", padx=20)
output_box = tk.Text(root, height=6, width=70, font=("Arial", 11),
                     bg="#313244", fg="#cdd6f4", state="disabled", relief="flat")
output_box.pack(padx=20, pady=5)

root.mainloop()
