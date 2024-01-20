#include <stdio.h>
#include <stdlib.h>//malloc函数所需头文件
int main()
{   
    // int number;

    // printf("输入数量：");
    // scanf("%d",&number);
    // // int a[number]; //only okay for C99
    // int *a;
    // a=(int*)malloc(number*sizeof(int));//malloc:交给a一块空间，后续就可以把a当数组使用
    // for(int i = 0;i<number;i++)
    // {
    //     scanf("%d",&a[i]);
    // }
    // free(a);//将空间返回；
    void *p;
    int cnt = 0;
    p=malloc(100*1024*1024);
    p++;
    free(p);
    while ((p=malloc(100*1024*1024)))//以字节为单位，故为100*1024byte*1024byte=100MB
    //一次申请100MB内存，若申请成功，则表达式（p=...）本身不为零，循环继续；反之，malloc返回0，循环退出
    {
        cnt++;
    }
    printf("分配了%d00MB的空间\n",cnt);
    system("pause"); 
    return 0;
}
