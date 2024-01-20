import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import keyboard
from threading import Thread
import datetime
from pystray import Icon as TrayIcon, MenuItem as item
import PIL.Image
import json
import os
import tkinter.font as tkFont


class EventManager:
    def __init__(self):
        self.event_list = []
        self.status_index = 0
        self.save_index = 0

    def do_event(self, cmd):
        pass

    def undo(self):
        if self.status_index > 0:
            undo_cmd = self.event_list[self.status_index - 1]
            self.status_index -= 1
            pass

    def redo(self):
        if self.status_index < len(self.event_list) - 1:
            redo_cmd = self.event_list[self.status_index + 1]
            self.status_index += 1
            pass

    def save(self):
        self.save_index = self.status_index
        pass


event_manager = EventManager()

def load_config():
    try:
        with open('config.json', 'r') as config_file:
            return json.load(config_file)
    except FileNotFoundError:
        return {}

def save_config(config):
    with open('config.json', 'w') as config_file:
        json.dump(config, config_file)

def toggle_window(window, text_box, key='ctrl+alt+a'):
    def on_key_event():
        if window.state() == 'normal':
            window.withdraw()
            window.attributes('-topmost', False)
        else:
            window.deiconify()
            window.attributes('-topmost', True)
            window.focus_force()
            text_box.focus_set()

    keyboard.add_hotkey(key, lambda: on_key_event())

def save_text(text_box, window):
    if hasattr(window, 'file_path') and window.file_path and os.path.exists(window.file_path):
        content = text_box.get('1.0', tk.END)
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(window.file_path, 'a', encoding='utf-8') as file:
            file.write(f'[{timestamp}]\n{content}\n')
        event_manager.save()
    else:
        messagebox.showerror("保存错误", "未选择保存路径或路径不存在")

def create_window():
    window = tk.Tk()
    window.title("神秘便签")
    code_font = tkFont.Font(family="Consolas", size=12)
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    window_width, window_height = 300, 200
    x = screen_width - window_width
    y = (screen_height - window_height) // 2
    window.geometry(f"{window_width}x{window_height}+{x}+{y}")

    control_frame = tk.Frame(window)
    control_frame.pack(side='top', fill='x')

    text_frame = tk.Frame(window)
    text_frame.pack(expand=True, fill='both')

    text_box = tk.Text(text_frame, undo=True, wrap='none')
    text_box.config(font=code_font)

    v_scrollbar = tk.Scrollbar(text_frame, orient='vertical', command=text_box.yview)
    v_scrollbar.pack(side='right', fill='y')
    text_box['yscrollcommand'] = v_scrollbar.set

    h_scrollbar = tk.Scrollbar(text_frame, orient='horizontal', command=text_box.xview)
    h_scrollbar.pack(side='bottom', fill='x')
    text_box['xscrollcommand'] = h_scrollbar.set

    text_box.pack(side='left', expand=True, fill='both')

    def zoom(event):
        nonlocal code_font
        current_font_size = code_font.actual()["size"]
        if event.delta > 0:
            current_font_size += 1
        else:
            current_font_size = max(1, current_font_size - 1)
        code_font.config(size=current_font_size)
        text_box.config(font=code_font)

    window.bind("<Control-MouseWheel>", zoom)

    def on_key_release(event):
        text_box.edit_separator()
    text_box.bind('<KeyRelease>', on_key_release)

    config = load_config()
    window.file_path = config.get('file_path')

    def select_file_path():
        window.file_path = filedialog.askopenfilename()
        save_config({'file_path': window.file_path})

    select_path_button = tk.Button(control_frame, text="选择路径", command=select_file_path)
    select_path_button.pack(side='left')

    def clear_text():
        text_box.delete('1.0', tk.END)

    clear_button = tk.Button(control_frame, text="清空", command=clear_text)
    clear_button.pack(side='left')

    toggle_window(window, text_box, 'ctrl+alt+a')

    window.bind('<Control-z>', lambda event: event_manager.undo())
    window.bind('<Control-Shift-Z>', lambda event: event_manager.redo())

    def setup_tray_icon():
        def on_show_window(icon, item):
            window.deiconify()
            window.attributes('-topmost', True)
            window.focus_force()
            text_box.focus_set()

        def on_exit(icon, item):
            icon.stop()
            window.destroy()

        image = PIL.Image.open("C:/DESKTOP/临时/长期用不到/图标/N.ico")
        menu = (item('显示', on_show_window), item('退出', on_exit))
        icon = TrayIcon("便签", image, "便签", menu)
        icon.run_detached()

        return icon

    icon_thread = Thread(target=setup_tray_icon)
    icon_thread.start()

    def on_close():
        if messagebox.askokcancel("退出", "确定要退出吗？"):
            if icon_thread.is_alive():
                icon_instance = icon_thread._target()
                icon_instance.stop()
            window.destroy()

    window.protocol("WM_DELETE_WINDOW", on_close)

    window.bind('<Control-s>', lambda event: save_text(text_box, window))

    window.mainloop()

create_window()
