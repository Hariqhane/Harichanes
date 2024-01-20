#include <stdio.h>
#include <stdlib.h>
int min(int,int,int);
int max(int,int,int);
int main()
{
    int x,y,z,mi_n,mid,ma_x;
    scanf("%d %d %d",&x,&y,&z);
    int arr[3]={x,y,z};
    getchar();
    mi_n = min(x,y,z);
    ma_x = max(x,y,z);
    for(int i=0;i<3;i++)
    {
        if(arr[i]!=mi_n&&arr[i]!=ma_x)
        {
            mid = arr[i];
            break;
        }
        
    }
    printf("%d %d %d",mi_n,mid,ma_x);
    getchar();
    return 0;
}
int min(int x,int y,int z)
{
        if(x<y)
    {
        if(x<z)
            return x;
        else
        {
            return z;
        }
    }
    else
    {
        if(y>z)
            return z;
        else 
            return y; 
    }

}
int max(int x,int y,int z)
{
    if(x>y)
    {
        if(x>z)
            return x;
        else
        {
            return z;
        }
    }
    else
    {
        if(y<z)
            return z;
        else 
            return y; 
    }
}

