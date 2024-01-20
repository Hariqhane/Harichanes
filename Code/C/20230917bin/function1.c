#include <stdio.h>
void sum(int start,int end )
{
int sum=0;
    int i;
    for (i=start;i<=end;i++){
        sum+=i;
    }
    printf("%d\n",sum);
}
int main()
{   
    int a,b;
    scanf("%d %d",&a,&b);
    sum(a,b);
    system("pause");
    return 0;
}