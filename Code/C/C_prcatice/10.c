#include <stdio.h>
#include <stdlib.h>
int main()
{
    int factor;
    for(int i = 1;i<=1000;i++)
    {
        factor = 0;
        for(int j = 1;j <i;j++)//不应该取等j<=i，因为不包括i本身
        {

            if(i%j == 0)
            {
                factor+=j;
            }
        
        }
        if(factor == i)
        {
            printf("%d\n",i);
        }
    }
    getchar();
    return 0;
}


