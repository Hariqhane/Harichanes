;;��CapsLk��ӳ��Ϊ����
;CapsLock::Backspace
;����Ϊalt+E��alt+����
; ����������м�ʱ���� Alt + E
MButton::Send, !e

; ������ Mouse4 ʱ���� Alt + �Ҽ�ͷ
XButton2::Send, !{Right}

; ������ Mouse5 ʱ���� Alt + ���ͷ
XButton1::Send, !{Left}

;����Ϊshift+����,����;ctrl+shift+����,���ٺ���
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
;CapsLk+C������Mathtype


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
            ; ������������Ӵ��봦����а����ݳ���С�ڵ���2����������籣��ԭ������ռ��а�
        }
    }
return

CapsLock & a::
	Send {Home}
	Send +{End}
Return
; ����CapsLock+W�İ������
; CapsLock & w::
;     ; ö�����д���
;     WinGet, windows, List
;     Loop, %windows%
;     {
;         ; ��ȡ���ھ��
;         windowId := windows%A_Index%
;         ; ���Թرմ���
;         WinClose, ahk_id %windowId%
;     }
; return
;win+C�����ļ���ַ
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

