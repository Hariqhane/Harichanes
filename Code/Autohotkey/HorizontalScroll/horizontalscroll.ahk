; !确保CapsLock本身的功能不会被触发
;SetCapsLockState, AlwaysOff
;以下为alt+E及alt+←→
; 当按下鼠标中键时发送 Alt + E
MButton::Send, !e

; 当按下 Mouse4 时发送 Alt + 右箭头
XButton2::Send, !{Right}

; 当按下 Mouse5 时发送 Alt + 左箭头
XButton1::Send, !{Left}

;!以下为shift+滚轮,横向;ctrl+shift+滚轮,加速横向
+WheelUp::  ; Scroll left.
    SetTitleMatchMode, 2
    IfWinActive, Excel
    {
        ;SetScrollLockState, on
        ComObjActive("Excel.Application").ActiveWindow.SmallScroll(0,0,0,2)
        ;SetScrollLockState, off
    }
    else IfWinActive, PowerPoint
    	ComObjActive("PowerPoint.Application").ActiveWindow.SmallScroll(0,0,0,3)
    else IfWinActive, Word
    	ComObjActive("Word.Application").ActiveWindow.SmallScroll(0,0,0,3)
    else IfWinActive, Adobe Acrobat Professional -
    {
        send,+{left}
    }
    else IfWinActive, - Mozilla Firefox
    {
		Loop 4
			send,{left}
    }
    else
    {
        ControlGetFocus, FocusedControl, A
		Loop 10
			SendMessage, 0x114, 0, 0, %FocusedControl%, A  ; 0x114 is WM_HSCROLL ; 1 vs. 0 causes SB_LINEDOWN vs. UP
    }
return


+WheelDown::  ; Scroll right.
    SetTitleMatchMode, 2
    IfWinActive, Excel
    {
        ;SetScrollLockState, on
        ComObjActive("Excel.Application").ActiveWindow.SmallScroll(0,0,2,0)
        ;SetScrollLockState, off
    }
    else IfWinActive, PowerPoint
    	ComObjActive("PowerPoint.Application").ActiveWindow.SmallScroll(0,0,3,0)
    else IfWinActive, Word
    	ComObjActive("Word.Application").ActiveWindow.SmallScroll(0,0,3,0)
    else IfWinActive, Adobe Acrobat Professional -
    {
        send,+{right}
    }
    else IfWinActive, - Mozilla Firefox
    {
		Loop 4
			send,{right}
    }
    else
    {
        ControlGetFocus, FocusedControl, A
		Loop 10
			SendMessage, 0x114, 1, 0, %FocusedControl%, A  ; 0x114 is WM_HSCROLL ; 1 vs. 0 causes SB_LINEDOWN vs. UP
    }
return

^+WheelUp::  ; Scroll left.
    SetTitleMatchMode, 2
    IfWinActive, Excel
    {
        ;SetScrollLockState, on
        ComObjActive("Excel.Application").ActiveWindow.SmallScroll(0,0,0,10)
        ;SetScrollLockState, off
    }
    else IfWinActive, PowerPoint
    	ComObjActive("PowerPoint.Application").ActiveWindow.SmallScroll(0,0,0,10)
    else IfWinActive, Word
    	ComObjActive("Word.Application").ActiveWindow.SmallScroll(0,0,0,10)
    else IfWinActive, Adobe Acrobat Professional -
    {
        send,+{left}
    }
    else IfWinActive, - Mozilla Firefox
    {
		Loop 4
			send,{left}
    }
    else
    {
        ControlGetFocus, FocusedControl, A
		Loop 500
			SendMessage, 0x114, 0, 0, %FocusedControl%, A  ; 0x114 is WM_HSCROLL ; 1 vs. 0 causes SB_LINEDOWN vs. UP ; ZHULAOJIANKEHAVEWRITTENITONAPRIL THETENTH INTWENTYTWENTYONE
    }
return

^+WheelDown::  ; Scroll right.
    SetTitleMatchMode, 2
    IfWinActive, Excel
    {
        ;SetScrollLockState, on
        ComObjActive("Excel.Application").ActiveWindow.SmallScroll(0,0,10,0)
        ;SetScrollLockState, off
    }
    else IfWinActive, PowerPoint
    	ComObjActive("PowerPoint.Application").ActiveWindow.SmallScroll(0,0,10,0)
    else IfWinActive, Word
    	ComObjActive("Word.Application").ActiveWindow.SmallScroll(0,0,10,0)
    else IfWinActive, Adobe Acrobat Professional -
    {
        send,+{right}
    }
    else IfWinActive, - Mozilla Firefox
    {
		Loop 4
			send,{right}
    }
    else
    {
        ControlGetFocus, FocusedControl, A
		Loop 100
			SendMessage, 0x114, 1, 0, %FocusedControl%, A  ; 0x114 is WM_HSCROLL ; 1 vs. 0 causes SB_LINEDOWN vs. UP
    }

return
;!CapsLk+C适用于Mathtype


; 当按下CapsLock+C时触发
CapsLock & c::
    IfWinActive, MathType ; 检查MathType窗口是否为活动窗口
    {
        Clipboard := "" ; 清空剪切板，为新内容做准备
        Send, ^c ; 模拟Ctrl+C复制选中的文本
        ClipWait, 1 ; 等待剪切板内容更新，最多等待1秒
        if (StrLen(Clipboard) > 2) ; 如果剪切板内容的长度大于2个字符
        {
            ; 从剪切板中删除第一个和最后一个字符
            Clipboard := SubStr(Clipboard, 2, StrLen(Clipboard) - 2)
        }
        else
        {
            ; 如果剪切板内容不足以删除两个字符，则不做任何改变
            ; 可以在这里添加代码处理剪切板内容长度小于等于2的情况，例如保持原样或清空剪切板
        }
    }
return
;!当前行全选
CapsLock & a::
	Send {Home}
	Send +{End}
Return
;!win+C复制文件地址
#c::
Clipboard =
Send,^c
ClipWait
path = %Clipboard%
Clipboard = %path%
Tooltip,%path%
Sleep,1000
Tooltip
Return
;!窗口置顶
; 监听CapsLock+~的按键组合
CapsLock & `::
    WinGet, ExStyle, ExStyle, A ; 获取当前活动窗口的扩展样式
    if (ExStyle & 0x8) { ; 检查是否已经设置为置顶窗口
        ; 如果已置顶，则取消置顶
        WinSet, AlwaysOnTop, Off, A
    } else {
        ; 如果未置顶，则设置为置顶
        WinSet, AlwaysOnTop, On, A
    }
return
;!win+L锁屏同时息屏
#L::  ; 按 Win + L 锁定电脑时触发
{
    Sleep 1000  ; 延迟1秒确保系统已经锁定
    SendMessage, 0x0112, 0xF170, 2, , Program Manager  ; 发送关闭屏幕的命令
    DllCall("LockWorkStation")  ; 调用系统锁定函数
}
return
