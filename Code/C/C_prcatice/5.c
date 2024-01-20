#include <stdio.h>
#include <stdlib.h>
int main()
{
    for(int i = 1;i<10;i++)
    {
        for(int j = 1;j<=i;j++)
        {
            if(i<j)
            printf("%d*%d=%-3d",i,j,i*j);/*-3d表示左对齐，占3位*/
            else
            printf("%d*%d=%-3d",j,i,i*j);
            if(j==i)
            printf("\n");
        }
    }
    getchar();
    return 0;
}


