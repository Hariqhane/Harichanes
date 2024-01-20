#include <stdio.h>
void minmax(int *,int len, int *max,int *min);
int main()
{   
    int a[]={1,2,2,5,5,5,9,49,4,-5,58,45};
    int len = sizeof(a)/sizeof(a[0]);
    int min,max;
    minmax(a,len,&max,&min);
    printf("%d,%d\n",min,max);
    system("pause"); 
    return 0;
}
void minmax(int a[],int len, int *max,int *min)
{
    *min = *max = a[0];
    for(int i = 1; i<len; i++)
    {
        if(a[i]<*min)
        {
            *min = a[i];//遍历即可实现
        }
        if(a[i]>*max)
        {
            *max = a[i];
        }
    }    

}