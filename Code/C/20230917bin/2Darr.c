#include <stdio.h>
int main()
{
//判断井字棋输赢
    const int size =3;
    int board[3][3];//1代表X，0代表O
    int i,j;
    int num0fx;//X的数量
    int num0f0;//O的数量
    int result =-1; //-1:no winner; 1:X win; 0: O win 
    //读入矩阵
    for(i=0;i<size;i++){
        for(j=0;j<size;j++){
            scanf("%d",&board[i][j]);//按行输入，从左到右
        }
    }
    //check solumn（行）
    for ( i = 0; i < size && result == -1; i++)
    {
        num0f0=num0fx=0;
        for( j = 0; j < size; j++){
            if (board[i][j] == 1){
                num0fx++;
            }
            else{
                num0f0++;
            }
        }
        if (num0f0 == size){
            result =0 ;
        }
        else if (num0fx ==size){
            result = 1;
        }
    // check line（列）
        else
        {
            num0f0=num0fx=0;
            for ( j = 0; j < size; j++)
            {
                for ( i = 0; i < size; i++)
                {
                    if (board[i][j]==1)
                    {
                        num0fx++;
                    }
                    else
                    {
                        num0f0++;
                    }
                    
                }
                if (num0f0 == size)
                {
                    result == 0;
                }
                else if(num0fx == size)
                {
                    result == 1;
                }
                //check  main diagnal（主对角线）
                else
                {
                    num0f0=num0fx=0;
                    for ( i = 0; i < size; i++)
                    {
                        if (board[i][i]==1)
                        {
                            num0fx++;
                        }
                        else
                        {
                            num0f0++;
                        }
                        
                    }
                    if (num0f0 == size)
                    {
                        result = 0;
                    }
                    else if (num0fx == size)
                    {
                        result = 1;
                    }
                    //check back-diagnal（反对角线）
                    else
                    {
                        for ( i = 0; i < size; i++)
                        {
                            if(board[i][size-1-i] == 1)
                                num0fx++;
                            else
                            {
                                num0f0++;
                            }
                        }
                        if (num0f0 == size)
                         {
                            result = 0;
                         }
                        else if (num0fx == size)
                         {
                            result = 1;                   
                         }
                }            
            }           
        }
    }
//将检查行与检查列合并：遍历行与列,
    int num1fo=0;
    int num1fx=0;
    for(i=0;i<size&&result == -1;i++)
    {
        num0f0=num0fx=num1fo=num1fx=0;
        for(j=0;j<size;j++)
        {
            if(board[i][j]==1)
            {
                num0fx++;
            }
            else if (board[i][j]==0)
            {
                num0f0++;
            }
            else if (board[j][i]==0)
            {
                num1fo++;
            }
            else
            {
                num1fx++;
            }
            if(num0f0==size||num1fo==size)
            {
                result=0;
            }
            else
            {
                result=1;
            }
        }
    }
    system("pause");
    return 0;
    }
}

//basic
    // int a[3][5]; //创建一个三行五列的矩阵
    // for(int i =0;i<3;i++){     //注意，同一维数组，每行、列为0~n-1
    //     for(int j = 0;j<5;j++){
    //         a[i][j] = i*j;    //a[i][j]表示第i行第j列，而不是a[i,j](这样等同于a[j])
    //         printf("%d\n",a[i][j]);
    //     }
    // }//这个双重for循环即为二维数组的遍历
//初始化
    // int a [][5]= {
    //     0,1,2,3,4,
    //     2,3,4,5,6,
    // };