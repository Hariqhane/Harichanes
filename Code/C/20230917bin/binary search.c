#include <stdio.h>
int search(int key ,int  a[], int len);
int main()
{
    int a[]={1,2,4,5,6,8,9,44,55,66,87,100,101};
    int len = sizeof(a)/sizeof(a[0]);
    int x;
    scanf("%d",&x);
    getchar();
    if(search(x,a,len)== -1){
        printf("该数组不包含此数字");
        }
         else{
            printf("该数组包含此数字，在第%d位",search(x,a,len)+1);
                }
    getchar();
    return 0;
}
int search(int key ,int  a[], int len)
{  
    int ret = -1;
    int left = 0;
    int right = len -1;
    while (right>=left)
    {
        int mid = (left +right )/2;
        if (a[mid] == key)
        {
            ret = mid;
            break;
        }else if (a[mid]>key)
        {
            right = mid -1;
        }else{
            left = mid + 1;
        }
    }
    return ret;
}