#include <stdio.h>
int search(int key , int a[] , int length);
int main(void)
{
    int a[]={1,2,3,4,5,6,7,8,9,10,11,11,13,};
    int x;
    int loc;
    printf("请输入一个数字:");
    scanf("%d",&x);
    getchar(); //代替system("pause"),因为后者只能在windows系统使用。
    loc=search(x,a,sizeof(a)/sizeof(a[0]));
    if(loc!=-1){
        printf("%d在第%d个位置上\n",x,loc);
    }
    else{
        printf("%d不存在\n",x);
    }
    getchar(); 
    return 0;
}
int search(int key,int a[],int length)
{
    int ret=-1;
    int i;
    for(i=0;i<length;i++){
        if(a[i]== key){
            ret =i;
            break;
        }
    }
    return ret;
}
