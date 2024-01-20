#include <stdio.h>
#include <stdlib.h>
int main()
{
    int i = 0;
    for(int x = 0;x<84;x++)
    {
        for(int i = x+1;i<84;i++)
        {
            if((i*i)==(x*x+168))//判断是否完全平方
            {
                printf("%d\n",x*x-100);
                break;
            }
        }
    }
    getchar();
    return 0;
}


