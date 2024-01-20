#include <stdio.h>
#include <stdlib.h>
int main()
{
    int x,y;
    int factor,mutiple;
    scanf("%d %d",&x,&y);
    getchar();
    for(int i = 1; i<=x&&i<=y;i++)
    {
        if(x%i==0&&y%i==0)
        {
            factor = i;
        }    
    }
    int i = x;  
    do{
        i++;
        mutiple = i;
    } while(i%x!=0||i%y!=0); 
    printf("这两个数的最大公因数是:%d\n",factor);
    printf("这两个数的最小公倍数是:%d\n",mutiple);
    getchar();
    return 0;
}


