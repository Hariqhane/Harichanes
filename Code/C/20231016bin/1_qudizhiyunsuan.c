#include <stdio.h>
int main()
{
    // int i = 0;
    // int p;
    // p = (int)&i;//强制类型转换，将变量地址转换为整型
    // printf("0x%x\n",p);//十六进制输出整型
    // printf("%p\n",&i);//输出将与上面不同
    // printf("%lu\n",sizeof(int));
    // printf("%lu\n",sizeof(&i));//64位架构下，int类型的数与地址长度不同
   
    // int i;
    // int p;
    // int l = 1;
    // printf("%p\n",&i);
    // printf("%p\n",&p);
    // printf("%p\n",&l);
//数组的地址
    int a[10];
    printf("%p\n",&a);
    printf("%p\n",a);
    printf("%p\n",&a[0]);
    printf("%p\n",&a[1]);
    getchar();
    return 0;
}