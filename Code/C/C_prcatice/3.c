#include <stdio.h>
#include <stdlib.h>
int main()
{
    const int big = 31;
    const int small = 30;
    const int february1 = 28;
    int y,m,d,day;
    printf("请输入年 月 日：");
    scanf("%d %d %d",&y,&m,&d);
    getchar();
    int m1 = big;
    int m2 = m1+february1;
    int m3 = m2+big;
    int m4 = m3+small;
    int m5 = m4+big;
    int m6 = m5+small;
    int m7 = m6+big;
    int m8 = m7+big;
    int m9 = m8+small;
    int m10 = m9+big;
    int m11 = m10+small;
    int m12 = m11+big;
    int a[13] ={0,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12};
    if(y%4!=0||m<3)
    {
        day = a[m-1]+d;
        printf("这一天是这一年的第%d天\n",day);
    }  
    else
    {
        day = a[m-1]+d+1;
    }
    printf("这一天是这一年的第%d天\n",day);
    getchar();
    return 0;
}


