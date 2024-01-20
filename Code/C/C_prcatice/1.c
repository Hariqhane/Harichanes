// 有 1、2、3、4 四个数字，
//能组成多少个互不相同且无重复数字的三位数？都是多少？
#include <stdio.h>
#include <stdlib.h>
int main()
{
    int cnt = 0;
    for(int i =1;i<=4;i++)
    {
        for(int j = 1; j<=4 ; j++)
        {
            for(int k = 1 ;k<=4; k++)
            {
                if(i==j||i==k||j==k)
                continue;
                cnt++;
                printf("第%d个数是:%d\n",cnt,100*i+10*j+k);
            }
        }
    }
    getchar();
    return 0;
}



