<?php
session_start();
?>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced driver safety system | Admin Login</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.6.2/dist/css/bootstrap.min.css">
  <script src="https://cdn.jsdelivr.net/npm/jquery@3.7.1/dist/jquery.slim.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/popper.js@1.16.1/dist/umd/popper.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@4.6.2/dist/js/bootstrap.bundle.min.js"></script>
<style>
    body
    {
        height:100vh;
        background-repeat:no-repeat;
        background-size:100% 100%;
        background-attachment:fixed;
        color:black;
        background-image:url("image/1.webp");
    }

</style>
</head>
<body>
    <h3 class="py-2 text-center text-white" style="background-color:rgba(0,0,0,0.7);">IOT Project</h3>
    <div class="container mt-4">
        <div class="row">
            <div class="col-md-6 offset-md-3">
                <div class="card mt-4" style="background-color:rgba(0,0,0,0.7);">
                    <div class="card-body">
                        <img src="image/1.png" style="height:240px;width:100%;border-radius:60px !important;">
                        <form action="" method="post" autocomplete="off">
                            
                          <div class="form-group">
                            <label for="email" class="text-white">Card number:</label>
                            <input type="number" class="form-control" name="card_no" placeholder="Enter Card number" id="email" required>
                          </div>
                          <div class="form-group">
                            <label for="pwd" class="text-white">Full name:</label>
                            <input type="text" class="form-control" name="full_name" placeholder="Enter full name" id="pwd" required>
                          </div>
                          <div class="row">
                              <div class="col">
                                  <div class="form-group">
                                    <label for="pwd" class="text-white">MM/YY:</label>
                                    <input type="text" class="form-control" name="month" placeholder="Enter MM/YY" id="pwd" required>
                                  </div>
                              </div>
                              <div class="col">
                                  <div class="form-group">
                                    <label for="pwd" class="text-white">CVC:</label>
                                    <input type="text" maxlength="3" class="form-control" name="cvc" placeholder="Enter CVC" id="pwd" required>
                                  </div>
                              </div>
                          </div>
                          <button type="submit" class="btn btn-primary btn-block" name="pay">Pay Now</button>
                        </form>
                    </div>
                </div>
            </div>
        </div>
    </div>
<?php
if(isset($_POST["pay"]))
{
   
    $card_no=$_POST["card_no"];
    $full_name=$_POST["full_name"];
    $month=$_POST["month"];
    $cvc=$_POST["cvc"];
    include("connection.php");
    $sql="insert into vp_fisherman values (NULL,'$card_no','$full_name','$month','$cvc',NULL)";
    
   
    if(mysqli_query($conn,$sql))
    {
        
        $_SESSION["fine"]="paid";
        $_SESSION["mode"]="";
        echo "<script>alert('fine paid');window.location.replace('index.php');</script>";
    }
    else
    {
        echo "<script>alert('Error to pay a fine');".mysqli_error($conn);
    }
}

?>
</body>
</html>
