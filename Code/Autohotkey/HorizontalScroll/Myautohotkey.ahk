; !ȷ��CapsLock�����Ĺ��ܲ��ᱻ����
SetCapsLockState, AlwaysOff
;����Ϊalt+E��alt+����
; ����������м�ʱ���� Alt + E
MButton::Send, !e

; ������ Mouse4 ʱ���� Alt + �Ҽ�ͷ
XButton2::Send, !{Right}

; ������ Mouse5 ʱ���� Alt + ���ͷ
XButton1::Send, !{Left}

;!����Ϊshift+����,����;ctrl+shift+����,���ٺ���
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
;!CapsLk+C������Mathtype


; ������CapsLock+Cʱ����
CapsLock & c::
    IfWinActive, MathType ; ���MathType�����Ƿ�Ϊ�����
    {
        Clipboard := "" ; ��ռ��а壬Ϊ��������׼��
        Send, ^c ; ģ��Ctrl+C����ѡ�е��ı�
        ClipWait, 1 ; �ȴ����а����ݸ��£����ȴ�1��
        if (StrLen(Clipboard) > 2) ; ������а����ݵĳ��ȴ���2���ַ�
        {
            ; �Ӽ��а���ɾ����һ�������һ���ַ�
            Clipboard := SubStr(Clipboard, 2, StrLen(Clipboard) - 2)
        }
        else
        {
            ; ������а����ݲ�����ɾ�������ַ��������κθı�
            ; �������������Ӵ��봦�����а����ݳ���С�ڵ���2����������籣��ԭ������ռ��а�
        }
    }
return
;!��ǰ��ȫѡ
CapsLock & a::
	Send {Home}
	Send +{End}
Return
;!win+C�����ļ���ַ
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
;!�����ö�
; ����CapsLock+~�İ������
CapsLock & `::
    WinGet, ExStyle, ExStyle, A ; ��ȡ��ǰ����ڵ���չ��ʽ
    if (ExStyle & 0x8) { ; ����Ƿ��Ѿ�����Ϊ�ö�����
        ; ������ö�����ȡ���ö�
        WinSet, AlwaysOnTop, Off, A
    } else {
        ; ���δ�ö���������Ϊ�ö�
        WinSet, AlwaysOnTop, On, A
    }
return
;!win+L����ͬʱϢ��
#L::  ; �� Win + L ��������ʱ����
{
    Sleep 500  ; �ӳ�1��ȷ��ϵͳ�Ѿ�����
    SendMessage, 0x0112, 0xF170, 2, , Program Manager  ; ���͹ر���Ļ������
    DllCall("LockWorkStation")  ; ����ϵͳ��������
}
return
;! ��;��'ȡ��Home��end
; ������Ctrl+Alt�������������Ϲ���ʱ��ҳ��������ƶ������
^!WheelUp::
    Send, {Home}
    return

; ������Ctrl+Alt�������������¹���ʱ��ҳ��������ƶ�����Ͷ�
^!WheelDown::
    Send, {End}
    return
; !����Alt+;ΪHome��
!;::
    Send {Home}
    return

; !����Alt+'ΪEnd��
!'::
    Send {End}
    return
; ������Ctrl+Alt+;ʱ������Ctrl+Home
^!;::
    Send ^{Home}
    return
; ������Ctrl+Alt+'ʱ������Ctrl+End
^!'::
    Send ^{End}
    return
; ������Ctrl+Shift+Alt+;ʱ������Ctrl+Shift+Home
^+!;::
    Send ^+{Home}
    return
; ������Ctrl+Shift+Alt+'ʱ������Ctrl+Shift+End
^+!'::
    Send ^+{End}
    return
; ������Shift+alt+;ʱ, ����Shift+Home
+!;::
    Send +{Home}
    return
; ������Shift+alt+;ʱ, ����Shift+End
+!'::
    Send +{End}
    return
;дC���Դ���ʱ, ����Ctrl+;,����β�ӷֺ�
^;:: ; Ctrl+; hotkey
Send, {End} ; Move cursor to the end of the line
Send, `{;}` ; Correctly type ;
return
^+Q:: ; 定义 Ctrl+Shift+Q 为热键
Send, {F2}
Sleep, 50 ; 稍等一下以确保操作顺畅进行
Send, {Right}
Sleep, 50 ; 稍等一下以确保操作顺畅进行
Send, {Backspace 7}
return
; 当按住Alt并向下滚动鼠标滚轮时，触发Win+T
!WheelDown::
SendInput, {LWin Down}t{LWin Up}
; Sleep, 50 ; 添加一个非常短的延迟
return
