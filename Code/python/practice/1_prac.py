import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import keyboard
from threading import Thread
import datetime
from pystray import Icon as TrayIcon, MenuItem as item
import PIL.Image
import json
import os


# 之前提供的EventManager类的代码
class EventManager:
    def __init__(self):
        self.event_list = []
        self.status_index = 0
        self.save_index = 0

    def do_event(self, cmd):
        # 执行真正的操作变化，这里需要根据你的应用逻辑进行修改
        pass

    def undo(self):
        if self.status_index > 0:
            # 获取撤回事件，依事件情况实现
            undo_cmd = self.get_undo_cmd(self.event_list[self.status_index])
            self.status_index -= 1
            # 执行真正的操作变化，这里需要根据你的应用逻辑进行修改
            pass

    def redo(self):
        if self.status_index < len(self.event_list) - 1:
            self.status_index += 1
            redo_cmd = self.event_list[self.status_index]
            # 执行真正的操作变化，这里需要根据你的应用逻辑进行修改
            pass

    def save(self):
        self.save_index = self.status_index
        # 重置软件的标题，这里需要根据你的应用逻辑进行修改
        pass

# 在你的应用中创建一个全局的EventManager对象
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
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    window_width, window_height = 300, 200
    x = screen_width - window_width
    y = (screen_height - window_height) // 2
    window.geometry(f"{window_width}x{window_height}+{x}+{y}")

    control_frame = tk.Frame(window)
    control_frame.pack(side='top', fill='x')

    text_box = tk.Text(window, undo=True)
    text_box.pack(expand=True, fill='both')

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

    def setup_tray_icon(window, text_box):
        def on_show_window(icon, item):
            window.deiconify()
            window.attributes('-topmost', True)
            window.focus_force()
            text_box.focus_set()

        def on_exit(icon, item):
            icon.stop()
            window.destroy()

        image = PIL.Image.open("C:/DESKTOP/临时/长期用不到/图标/favicon.ico")
        menu = (item('显示', on_show_window), item('退出', on_exit))
        icon = TrayIcon("便签", image, "便签", menu)
        icon.run_detached()

    Thread(target=setup_tray_icon, args=(window, text_box)).start()

    window.bind('<Control-s>', lambda event: save_text(text_box, window))

    window.mainloop()

create_window()
