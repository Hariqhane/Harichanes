#include <stdio.h>
int main()
{
    int x;
    int codd=0;
    int ceven=0;
    scanf("%d",&x);
    while(x!=-1){
        
        if(x%2==0){
            ceven+=1;
        }
        else{
            codd+=1;
        }
        scanf("%d",&x);
    }
    printf("奇数个数为：%d，偶数个数为：%d",codd,ceven);
    system("pause");
    return 0;
}