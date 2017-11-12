<!DOCTYPE html>
<html>
    <head>
        <title>SentiWorldNet</title>
        <link href="https://fonts.googleapis.com/css?family=Acme" rel="stylesheet">
        <style>
            body {
                background-image: url(background.jpg);
                color: floralwhite;
            }
            header .heading
            {
                font-family: 'Acme', sans-serif;
                font-size: 300%;
            }    
        </style>
    </head>
    <body>
        <header>
            <h1 class='heading'><center>Senti World Net Results</center></h1> 
        </header>
        <center>
        <?php
        $pyscript="senti.py";
        $cmd="$pyscript";
        $sentiaccuracy=exec("$cmd");
        echo $sentiaccuracy;
        ?>
        <br>
        <br>
        </center>
        <?php
        $pyscript="sentiAccArray.py";
        $cmd="$pyscript";
        $pred_array=explode(',',exec("$cmd"));
        #print_r($pred_array);
        #$varr=$pred_array[0];
        #echo $varr;
        ?>
        <?php
            $row=0;
            if(($handle = fopen("testdata.manual.2009.06.14.csv","r"))!==FALSE){
                $wantedColumns=array(5);
                while(($data=fgetcsv($handle))!==FALSE)
                {
                    $num=count($data);
                    for($c=0;$c<$num;$c++){
                        if(!in_array($c,$wantedColumns)) continue;
                        if(empty($data[$c])){
                            $value="&nbsp;";
                        }
                        else{
                            $value=$data[$c];
                        }
                        $varr=$pred_array[$row];
                            echo '<tr><td style="border-top: 1px solid rgb(111,180,224); border-left: 1px solid rgb(111,180,224); border-bottom: 1px solid rgb(111,180,224);"  align="left" bgcolor="#0066cc" height="36" valign="middle" ><b><font color="#ffffff" size="2">&nbsp;&nbsp;'.$value.'&nbsp;&nbsp;</font></b></td>';
                            echo '<td>';
                            echo $varr;
                            echo '</td><br>';
                        $row++;
                    }
                }
                echo '</tbody></table>';
                echo '</center>';   
                fclose($handle);
                }
        ?>
    </body>
</html>