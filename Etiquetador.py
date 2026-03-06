#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 12:27:48 2026

@author: Jan Beeck
"""

"""
=============================================================
  Anotador Manual de Bounding Boxes  ─  Exporta formato YOLO
  Usa SAM 3 (Ultralytics) para refinar máscaras (opcional)
=============================================================

Requisitos:
    pip install pillow ultralytics

Uso:
    python anotador_yolo.py                    # abre diálogo para elegir imagen
    python anotador_yolo.py ruta/imagen.jpg    # carga imagen directamente

Controles:
    Clic + arrastrar   → dibuja bounding box
    Enter / Botón OK   → confirma etiqueta
    Ctrl+Z             → deshacer último box
    Supr               → eliminar box seleccionado
    Ctrl+S             → guardar anotaciones YOLO
    Esc                → salir
=============================================================
"""

import sys
import os
import json
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from PIL import Image, ImageTk

# ─── Colores por clase ────────────────────────────────────────────────────────
PALETTE = [
    "#FF4757", "#2ED573", "#1E90FF", "#FF6348", "#ECCC68",
    "#A29BFE", "#00CEC9", "#FD79A8", "#FDCB6E", "#6C5CE7",
    "#FF3F34", "#05C46B", "#575FCF", "#EF5777", "#0be881",
]

# ─── Utilidad: color por nombre de clase ─────────────────────────────────────
_class_color_cache: dict[str, str] = {}

def color_for_class(label: str, all_classes: list[str]) -> str:
    if label not in _class_color_cache:
        idx = all_classes.index(label) if label in all_classes else len(all_classes)
        _class_color_cache[label] = PALETTE[idx % len(PALETTE)]
    return _class_color_cache[label]


# ─── Modelo SAM 3 (carga bajo demanda) ───────────────────────────────────────
_sam_model = None

def get_sam():
    global _sam_model
    if _sam_model is None:
        try:
            from ultralytics import SAM
            print("Cargando SAM 3 … (solo la primera vez)")
            _sam_model = SAM("sam3.pt")
            print("SAM 3 listo.")
        except Exception as e:
            messagebox.showwarning(
                "SAM 3 no disponible",
                f"No se pudo cargar SAM 3:\n{e}\n\n"
                "Puedes seguir anotando sin refinamiento.",
            )
    return _sam_model


def refine_with_sam(image_path: str, x1: int, y1: int, x2: int, y2: int):
    """
    Usa SAM 3 con un bounding-box prompt para obtener la máscara precisa
    del objeto. Devuelve (x1, y1, x2, y2) ajustados al bounding-box de la
    máscara, o los valores originales si SAM no está disponible.
    """
    model = get_sam()
    if model is None:
        return x1, y1, x2, y2
    try:
        results = model.predict(
            source=image_path,
            bboxes=[[x1, y1, x2, y2]],
            save=False,
            verbose=False,
        )
        r = results[0]
        if r.masks is not None and len(r.masks.data) > 0:
            import numpy as np
            mask = r.masks.data[0].cpu().numpy()
            img  = Image.open(image_path)
            mask_resized = np.array(
                Image.fromarray((mask * 255).astype("uint8")).resize(img.size[::-1])  # HxW
            )
            ys, xs = np.where(mask_resized > 127)
            if len(xs) > 0:
                return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    except Exception as e:
        print(f"SAM refinement error: {e}")
    return x1, y1, x2, y2


# ─── Aplicación principal ─────────────────────────────────────────────────────
class Annotator(tk.Tk):

    MAX_DISPLAY = 900   # px máx de ancho/alto en pantalla

    def __init__(self, image_path: str | None = None):
        super().__init__()
        self.title("Anotador YOLO · SAM 3")
        self.resizable(True, True)
        self.configure(bg="#111827")

        # Estado
        self.image_path: str | None = None
        self.orig_img:  Image.Image | None = None
        self.tk_img:    ImageTk.PhotoImage | None = None
        self.scale:     float = 1.0

        self.boxes:    list[dict] = []   # {label, x1, y1, x2, y2}  coords reales
        self.classes:  list[str] = []

        self._drag_start: tuple[int, int] | None = None
        self._rect_id:    int | None = None
        self._selected:   int | None = None  # índice del box seleccionado

        self._build_ui()
        self._bind_keys()

        if image_path and os.path.isfile(image_path):
            self._load_image(image_path)
        else:
            self.after(100, self._open_image)

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── barra superior ──────────────────────────────────────────────────
        top = tk.Frame(self, bg="#1f2937", pady=6, padx=10)
        top.pack(fill=tk.X, side=tk.TOP)

        tk.Label(top, text="Etiqueta:", bg="#1f2937", fg="#9ca3af",
                 font=("Segoe UI", 10)).pack(side=tk.LEFT, padx=(0, 4))

        self.label_var = tk.StringVar()
        self.label_entry = tk.Entry(
            top, textvariable=self.label_var, width=18,
            bg="#374151", fg="#f9fafb", insertbackground="white",
            relief=tk.FLAT, font=("Segoe UI", 10),
        )
        self.label_entry.pack(side=tk.LEFT, padx=(0, 6), ipady=3)
        self.label_entry.bind("<Return>", lambda e: self._confirm_label())

        # Autocompletado: menú de clases existentes
        self.class_menu = ttk.Combobox(
            top, textvariable=self.label_var, values=[], width=14,
            font=("Segoe UI", 10),
        )
        self.class_menu.pack(side=tk.LEFT, padx=(0, 10))
        self.class_menu.bind("<<ComboboxSelected>>", lambda e: self.label_entry.focus())

        self._btn(top, "✔ Agregar",      "#7c3aed", self._confirm_label).pack(side=tk.LEFT, padx=2)
        self._btn(top, "↩ Deshacer",     "#374151", self._undo).pack(side=tk.LEFT, padx=2)
        self._btn(top, "🗑 Eliminar sel.", "#dc2626", self._delete_selected).pack(side=tk.LEFT, padx=2)
        self._btn(top, "⚡ Refinar SAM3", "#0891b2", self._refine_selected).pack(side=tk.LEFT, padx=2)
        self._btn(top, "💾 Guardar YOLO", "#059669", self._save_yolo).pack(side=tk.LEFT, padx=2)
        self._btn(top, "📂 Abrir imagen", "#4b5563", self._open_image).pack(side=tk.LEFT, padx=2)

        self.status_var = tk.StringVar(value="0 boxes")
        tk.Label(top, textvariable=self.status_var, bg="#1f2937", fg="#6ee7b7",
                 font=("Segoe UI", 10)).pack(side=tk.RIGHT, padx=10)

        # ── área central: canvas + lista ────────────────────────────────────
        middle = tk.Frame(self, bg="#111827")
        middle.pack(fill=tk.BOTH, expand=True)

        # Canvas con scrollbars
        canvas_frame = tk.Frame(middle, bg="#111827")
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(
            canvas_frame, bg="#1f2937", cursor="crosshair",
            highlightthickness=0,
        )
        hbar = tk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL,
                            command=self.canvas.xview)
        vbar = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL,
                            command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        hbar.pack(side=tk.BOTTOM, fill=tk.X)
        vbar.pack(side=tk.RIGHT,  fill=tk.Y)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>",   self._on_press)
        self.canvas.bind("<B1-Motion>",       self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<ButtonPress-3>",   self._on_right_click)

        # ── panel derecho: lista de boxes ───────────────────────────────────
        right = tk.Frame(middle, bg="#1f2937", width=260)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)

        tk.Label(right, text="BOUNDING BOXES", bg="#1f2937", fg="#6b7280",
                 font=("Segoe UI", 9, "bold")).pack(pady=(8, 4))

        self.listbox = tk.Listbox(
            right, bg="#111827", fg="#e5e7eb", selectbackground="#7c3aed",
            font=("Consolas", 9), relief=tk.FLAT, bd=0,
            activestyle="none",
        )
        self.listbox.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))
        self.listbox.bind("<<ListboxSelect>>", self._on_list_select)

        # ── barra inferior ───────────────────────────────────────────────────
        bot = tk.Frame(self, bg="#0f172a", pady=4)
        bot.pack(fill=tk.X, side=tk.BOTTOM)
        self.hint_var = tk.StringVar(
            value="Arrastra sobre la imagen para dibujar un box · Clic derecho sobre un box para editar etiqueta"
        )
        tk.Label(bot, textvariable=self.hint_var, bg="#0f172a", fg="#475569",
                 font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=10)

    def _btn(self, parent, text, color, cmd):
        return tk.Button(
            parent, text=text, bg=color, fg="white", activebackground=color,
            activeforeground="white", relief=tk.FLAT, font=("Segoe UI", 9, "bold"),
            padx=8, pady=3, cursor="hand2", command=cmd,
        )

    def _bind_keys(self):
        self.bind("<Control-z>", lambda e: self._undo())
        self.bind("<Control-s>", lambda e: self._save_yolo())
        self.bind("<Delete>",    lambda e: self._delete_selected())
        self.bind("<Escape>",    lambda e: self.destroy())
        self.bind("<Return>",    lambda e: self._confirm_label())

    # ── Carga de imagen ───────────────────────────────────────────────────────
    def _open_image(self):
        path = filedialog.askopenfilename(
            title="Selecciona una imagen",
            filetypes=[
                ("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                ("Todos",    "*.*"),
            ],
        )
        if path:
            self._load_image(path)

    def _load_image(self, path: str):
        self.image_path = path
        self.orig_img   = Image.open(path).convert("RGB")
        self.boxes      = []
        self.classes    = []
        _class_color_cache.clear()
        self._selected  = None

        W, H = self.orig_img.size
        max_dim = self.MAX_DISPLAY
        self.scale = min(max_dim / W, max_dim / H, 1.0)

        dw = int(W * self.scale)
        dh = int(H * self.scale)
        display_img = self.orig_img.resize((dw, dh), Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(display_img)

        self.canvas.config(width=dw, height=dh,
                           scrollregion=(0, 0, dw, dh))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img, tags="bg")

        self.title(f"Anotador YOLO · {os.path.basename(path)}  ({W}×{H} px)")
        self._refresh_list()
        self.hint_var.set(
            f"Imagen: {os.path.basename(path)}  {W}×{H} px  "
            f"(escala {self.scale:.2f}) · Arrastra para dibujar"
        )

    # ── Eventos de ratón ──────────────────────────────────────────────────────
    def _canvas_to_real(self, cx, cy):
        """Convierte coordenadas canvas → coordenadas imagen original."""
        return int(cx / self.scale), int(cy / self.scale)

    def _on_press(self, event):
        self._drag_start = (event.x, event.y)
        if self._rect_id:
            self.canvas.delete(self._rect_id)
        self._rect_id = None

    def _on_drag(self, event):
        if not self._drag_start:
            return
        x0, y0 = self._drag_start
        if self._rect_id:
            self.canvas.delete(self._rect_id)
        self._rect_id = self.canvas.create_rectangle(
            x0, y0, event.x, event.y,
            outline="#facc15", width=2, dash=(5, 3),
        )

    def _on_release(self, event):
        if not self._drag_start:
            return
        x0, y0 = self._drag_start
        x1, y1 = event.x, event.y
        self._drag_start = None

        if abs(x1 - x0) < 5 or abs(y1 - y0) < 5:
            if self._rect_id:
                self.canvas.delete(self._rect_id)
                self._rect_id = None
            return

        # Guardar coordenadas del box pendiente en real
        rx0, ry0 = self._canvas_to_real(min(x0, x1), min(y0, y1))
        rx1, ry1 = self._canvas_to_real(max(x0, x1), max(y0, y1))
        self._pending_box = (rx0, ry0, rx1, ry1)

        # Pedir etiqueta: si ya hay texto en el entry, usarlo directamente
        lbl = self.label_var.get().strip()
        if lbl:
            self._confirm_label()
        else:
            self.label_entry.focus_set()
            self.hint_var.set("Escribe la etiqueta y pulsa Enter o haz clic en Agregar")

    def _confirm_label(self):
        lbl = self.label_var.get().strip()
        if not lbl:
            messagebox.showwarning("Etiqueta vacía", "Escribe un nombre de etiqueta primero.")
            return
        if not hasattr(self, "_pending_box") or self._pending_box is None:
            messagebox.showwarning("Sin box", "Dibuja un bounding box primero arrastrando sobre la imagen.")
            return

        x1, y1, x2, y2 = self._pending_box
        self._pending_box = None

        if self._rect_id:
            self.canvas.delete(self._rect_id)
            self._rect_id = None

        self._add_box(lbl, x1, y1, x2, y2)
        self.hint_var.set("Box agregado. Sigue dibujando o guarda con Ctrl+S.")

    def _on_right_click(self, event):
        """Clic derecho sobre un box dibujado → editar etiqueta."""
        rx, ry = self._canvas_to_real(event.x, event.y)
        for i, b in enumerate(self.boxes):
            if b["x1"] <= rx <= b["x2"] and b["y1"] <= ry <= b["y2"]:
                new_lbl = simpledialog.askstring(
                    "Editar etiqueta", f"Etiqueta actual: {b['label']}\nNueva etiqueta:",
                    initialvalue=b["label"], parent=self,
                )
                if new_lbl and new_lbl.strip():
                    self.boxes[i]["label"] = new_lbl.strip()
                    self._sync_classes()
                    self._redraw_boxes()
                    self._refresh_list()
                break

    # ── Gestión de boxes ──────────────────────────────────────────────────────
    def _add_box(self, label: str, x1: int, y1: int, x2: int, y2: int):
        if label not in self.classes:
            self.classes.append(label)
            self.class_menu["values"] = self.classes

        self.boxes.append({"label": label, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
        self._redraw_boxes()
        self._refresh_list()

    def _undo(self):
        if self.boxes:
            self.boxes.pop()
            self._sync_classes()
            self._redraw_boxes()
            self._refresh_list()

    def _delete_selected(self):
        if self._selected is not None and 0 <= self._selected < len(self.boxes):
            self.boxes.pop(self._selected)
            self._selected = None
            self._sync_classes()
            self._redraw_boxes()
            self._refresh_list()

    def _sync_classes(self):
        """Recalcula self.classes basado en los boxes actuales."""
        seen = []
        for b in self.boxes:
            if b["label"] not in seen:
                seen.append(b["label"])
        self.classes = seen
        self.class_menu["values"] = self.classes

    # ── Refinamiento con SAM 3 ────────────────────────────────────────────────
    def _refine_selected(self):
        if self._selected is None:
            messagebox.showinfo("Selecciona un box", "Haz clic en un box de la lista para seleccionarlo.")
            return
        if not self.image_path:
            return
        b = self.boxes[self._selected]
        self.hint_var.set("Refinando con SAM 3 … por favor espera")
        self.update()
        nx1, ny1, nx2, ny2 = refine_with_sam(
            self.image_path, b["x1"], b["y1"], b["x2"], b["y2"]
        )
        self.boxes[self._selected].update(x1=nx1, y1=ny1, x2=nx2, y2=ny2)
        self._redraw_boxes()
        self._refresh_list()
        self.hint_var.set(f"Box refinado con SAM 3: ({nx1},{ny1}) → ({nx2},{ny2})")

    # ── Dibujo en canvas ──────────────────────────────────────────────────────
    def _redraw_boxes(self):
        self.canvas.delete("box")
        for i, b in enumerate(self.boxes):
            col = color_for_class(b["label"], self.classes)
            selected = (i == self._selected)

            x1 = b["x1"] * self.scale
            y1 = b["y1"] * self.scale
            x2 = b["x2"] * self.scale
            y2 = b["y2"] * self.scale

            # Rectángulo
            self.canvas.create_rectangle(
                x1, y1, x2, y2,
                outline=col, width=3 if selected else 2,
                dash=() if not selected else (8, 4),
                tags="box",
            )

            # Etiqueta
            txt = f"[{i}] {b['label']}"
            tx, ty = x1, y1 - 18 if y1 > 18 else y2
            self.canvas.create_rectangle(
                tx, ty, tx + len(txt) * 7 + 8, ty + 18,
                fill=col, outline="", tags="box",
            )
            self.canvas.create_text(
                tx + 4, ty + 9, text=txt, fill="white",
                font=("Segoe UI", 9, "bold"), anchor=tk.W, tags="box",
            )

    # ── Lista lateral ─────────────────────────────────────────────────────────
    def _refresh_list(self):
        self.listbox.delete(0, tk.END)
        for i, b in enumerate(self.boxes):
            self.listbox.insert(
                tk.END,
                f"[{i:02d}] {b['label']:<16} "
                f"({b['x1']},{b['y1']})→({b['x2']},{b['y2']})",
            )
        self.status_var.set(f"{len(self.boxes)} boxes · {len(self.classes)} clases")

    def _on_list_select(self, _event):
        sel = self.listbox.curselection()
        if sel:
            self._selected = sel[0]
            self._redraw_boxes()

    # ── Guardar YOLO ──────────────────────────────────────────────────────────
    def _save_yolo(self):
        if not self.boxes:
            messagebox.showwarning("Sin anotaciones", "No hay bounding boxes para guardar.")
            return
        if not self.image_path:
            return

        W, H = self.orig_img.size
        base  = os.path.splitext(self.image_path)[0]

        # ── .txt YOLO ────────────────────────────────────────────────────────
        lines = []
        for b in self.boxes:
            cid = self.classes.index(b["label"])
            cx  = round((b["x1"] + b["x2"]) / 2 / W, 6)
            cy  = round((b["y1"] + b["y2"]) / 2 / H, 6)
            bw  = round((b["x2"] - b["x1"]) / W,     6)
            bh  = round((b["y2"] - b["y1"]) / H,     6)
            lines.append(f"{cid} {cx} {cy} {bw} {bh}")

        ann_path = base + ".txt"
        with open(ann_path, "w") as f:
            f.write("\n".join(lines))

        # ── classes.txt ──────────────────────────────────────────────────────
        cls_path = os.path.join(os.path.dirname(self.image_path), "classes.txt")
        with open(cls_path, "w") as f:
            f.write("\n".join(self.classes))

        # ── JSON de respaldo ─────────────────────────────────────────────────
        json_path = base + "_annotations.json"
        with open(json_path, "w") as f:
            json.dump(
                {"image": self.image_path, "size": [W, H],
                 "classes": self.classes, "boxes": self.boxes},
                f, indent=2,
            )

        msg = (
            f"✅ Archivos guardados:\n\n"
            f"  {ann_path}\n"
            f"  {cls_path}\n"
            f"  {json_path}\n\n"
            f"Boxes: {len(lines)}\n"
            f"Clases: {self.classes}"
        )
        messagebox.showinfo("Guardado", msg)
        self.hint_var.set(f"Guardado → {ann_path}")
        print("\n" + msg)


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) > 1 else None
    app = Annotator(image_path=img_path)
    app.mainloop()
