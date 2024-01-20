#include <stdio.h>
#include <stdlib.h>
int max(int x, int y)
{
    return x > y ? x : y;
}
 
int main(void)
{
    int (* p)(int, int) = & max; // &可以省略
    int a, b, c, d;
 
    printf("请输入三个数字:");
    scanf("%d %d %d", & a, & b, & c);
    d = p(p(a, b), c);//与直接调用函数等价，d = max(max(a, b), c) 
    printf("最大的数字是: %d\n", d);    getchar();
    return 0;
}


