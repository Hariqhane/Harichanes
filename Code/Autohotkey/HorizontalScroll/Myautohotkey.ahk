; 禁用CapsLock键
SetCapsLockState, AlwaysOff

; 鼠标中键模拟Alt+E快捷键
MButton::Send, !e

; 鼠标侧键模拟浏览器前进后退
XButton2::Send, !{Right}
XButton1::Send, !{Left}
; 滚轮上滚进行特定应用的滚动或其他动作
+WheelUp::
SetTitleMatchMode, 2
IfWinActive, Excel
{
    ComObjActive("Excel.Application").ActiveWindow.SmallScroll(0,0,0,2)
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
        SendMessage, 0x114, 0, 0, %FocusedControl%, A
}
return
; 滚轮下滚进行特定应用的滚动或其他动作
+WheelDown::
SetTitleMatchMode, 2
IfWinActive, Excel
{
    ComObjActive("Excel.Application").ActiveWindow.SmallScroll(0,0,2,0)
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
        SendMessage, 0x114, 1, 0, %FocusedControl%, A
}
return


; Ctrl+Shift+滚轮上滚在特定应用中进行更大幅度的滚动
^+WheelUp::
SetTitleMatchMode, 2
IfWinActive, Excel
{
    ComObjActive("Excel.Application").ActiveWindow.SmallScroll(0,0,0,10)
}
else IfWinActive, PowerPoint
    ComObjActive("PowerPoint.Application").ActiveWindow.SmallScroll(0,0,0,10)
else IfWinActive, Word
    ComObjActive("Word.Application").ActiveWindow.SmallScroll(0,0,0,10)
else
{
    ControlGetFocus, FocusedControl, A
    Loop 500
        SendMessage, 0x114, 0, 0, %FocusedControl%, A
}
return

; Ctrl+Shift+滚轮下滚在特定应用中进行更大幅度的滚动
^+WheelDown::
SetTitleMatchMode, 2
IfWinActive, Excel
{
    ComObjActive("Excel.Application").ActiveWindow.SmallScroll(0,0,10,0)
}
else IfWinActive, PowerPoint
    ComObjActive("PowerPoint.Application").ActiveWindow.SmallScroll(0,0,10,0)
else IfWinActive, Word
    ComObjActive("Word.Application").ActiveWindow.SmallScroll(0,0,10,0)
else
{
    ControlGetFocus, FocusedControl, A
    Loop 100
        SendMessage, 0x114, 1, 0, %FocusedControl%, A
}
return

; 使用CapsLock+c复制MathType公式
CapsLock & c::
IfWinActive, MathType
{
    Clipboard := ""
    Send, ^c
    ClipWait, 1
    if (StrLen(Clipboard) > 2)
    {
        Clipboard := SubStr(Clipboard, 2, StrLen(Clipboard) - 2)
    }
}
return

; CapsLock+a快捷键选取当前行文本
CapsLock & a::
Send {Home}
Send +{End}
Return

; Ctrl+CapsLock+c复制文件路径到剪切板并显示为提示信息
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

; CapsLock+`切换当前窗口的置顶状态
CapsLock & `::
WinGet, ExStyle, ExStyle, A
if (ExStyle & 0x8) {
    WinSet, AlwaysOnTop, Off, A
} else {
    WinSet, AlwaysOnTop, On, A
}
return

; #L锁定计算机
#L::
{
    Sleep 500  
    SendMessage, 0x0112, 0xF170, 2,, Program Manager  
    DllCall("LockWorkStation")  
}
return

; #Ctrl+Alt+Scroll, Home&end
^!WheelUp::
    Send, {Home}
    return
^!WheelDown::
    Send, {End}
    return

; Alt+; Home
!;::
    Send {Home}
    return

;ALt+', End
!'::
    Send {End}
    return
; ^: ctrl; !: alt; 
^!;::
    Send ^{Home}
    return
^!'::
    Send ^{End}
    return

^+!;::
    Send ^+{Home}
    return
^+!'::
    Send ^+{End}
    return
+!;::
    Send +{Home}
    return
+!'::
    Send +{End}
    return

; Ctrl+; , 在行尾添加;
^;:: ; Ctrl+; hotkey
    Send, {End} ; Move cursor to the end of the line
    Send, `{;}` ; Correctly type ;
    return
^+;::
    Send, {End} ; Move cursor to the end of the line
    Send, `{:}` ; Correctly type ;
    return
; Ctrl+Shift+Q, 在稻壳阅读器中添加书签时自动删除后七个字符
^+Q:: 
    Send, {F2}
    Sleep, 50 ; 稍等一下以确保操作顺畅进行
    Send, {Right}
    Sleep, 50 ; 稍等一下以确保操作顺畅进行
    Send, {Backspace 7}
    return

; 当按住Alt并向下滚动鼠标滚轮时，触发Win+T
!WheelDown::
    SendInput, {LWin Down}t{LWin Up}
    return

; 定义 Ctrl+Shift+W 删除快捷方式后缀 
^+W:: 
    Send, {F2}
    Sleep, 50 ; 稍等一下以确保操作顺畅进行
    Send, {Right}
    Sleep, 50 ; 稍等一下以确保操作顺畅进行
    Send, {Backspace 11}
    return

; 在Onenote中的上下角标快捷键, 用Ctrl+H和Ctrl和Ctrl+L
#IfWinActive ahk_class Framework::CFrame
^H:: ; 当按下Ctrl+H时
    Send, ^+{=} ; 模拟按下Ctrl+Shift+=
    return

^L:: ; 当按下Ctrl+L时
    Send, ^{=} ; 模拟按下Ctrl+=
    return

#IfWinActive
