#include <stdio.h>
int judgehave(int a[],int len,int x)
{
    int judgement = -1;
    for ( int count = 0; count<len; count++)
    {
        if(x== a[count]){
            judgement= 1;
            printf("该数组存在该数字，在第%d位",count);
            break;
        }
    }
    if (judgement == -1)
    {
        printf("该数组不存在该数字");
    }
}
int main(void)
{
    const int number = 100;
    int a[number];
    for ( int i = 0; i < number; i++)
    {
        a[i]=i+100;
    }
    int x;
    scanf("%d",&x);
    judgehave(a,sizeof(a)/sizeof(a[0]),x);
    getchar();
    getchar();
    return 0;
}
