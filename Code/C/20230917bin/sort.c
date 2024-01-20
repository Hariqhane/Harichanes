#include <stdio.h>
//step 1:找到数组中最大数所在位置
//step 2:swap a[maxid],a[len-1]
int max(int a[],int len)
{
    int maxid= 0;
    for(int i = 1; i<len;i++)
    {
        if (a[i]>a[maxid])
        {
            maxid = i;
        }
    }
    return maxid;
}//找到数组中最大数所在位置
int main()
{
    int a[]={5,12,16,56,88,1,22,22,13,25,42};
    int len = sizeof(a)/sizeof(a[0]);
    for(int i = 1; i < len;i++)
    {
        int maxid = max(a,len-i);
        int t=a[maxid];
        a[maxid]=a[len-i];
        a[len-i]=t;
    }
    for(int count = 0;count<len;count++)
    {
        printf("%d\n",a[count]);
    }
    getchar();
    return 0;
}
