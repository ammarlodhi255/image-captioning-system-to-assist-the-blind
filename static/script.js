const dropArea = document.querySelector(".drag-area")
dragText = dropArea.querySelector("header")
button = dropArea.querySelector("button")
input = dropArea.querySelector("input")
da = document.querySelector(".drag-area").innerHTML
let file; 
removeImageButton = document.getElementById('rmv-img-btn')

function browseBtnCLicked() { 
  document.getElementById('input').click(); 
}


// function speak() {
//   var audio = new Audio('D:\\University Files\\Assignments\\7th Semester\\Machine Learning\\Project\\source-code\\static\\speech.wav')
//   audio.play()
// }


function inputChanged() {
  //getting user select file and [0] this means if user selecst multiple files then we'll select only the first one
  file = document.getElementById('input').files[0];
  dropArea.classList.add("active");
  showFile(); 
}


//If user Drags File Over DropArea
function dragOver(event) {
  event.preventDefault(); //preventing from default behaviour
  dropArea.classList.add("active");
  dropArea.querySelector("header").textContent = "Release to Upload File";
}

//If user leaves dragged File from DropArea
function dragLeave() {
  dropArea.classList.remove("active");
  dropArea.querySelector("header").textContent = "Drag & Drop to Upload File";
}


function dropped(event) {
  event.preventDefault(); //preventing from default behaviour
  //getting user select file and [0] this means if user select multiple files then we'll select only the first one
  file = event.dataTransfer.files[0];
  showFile(); 
}

function showFile() {
  document.getElementById('caption').innerText = ''
  let fileType = file.type; 
  let validExtensions = ["image/jpeg", "image/jpg", "image/png"]; 
  
  if(validExtensions.includes(fileType)) { 
    let fileReader = new FileReader(); 
    fileReader.onload = () => {
      //passing user file source in fileURL variable
      let fileURL = fileReader.result; 
      //creating an img tag and passing user selected file source inside src attribute
      let imgTag = `<img src="${fileURL}" alt="image" class="imageDiv">`; 
      dropArea.innerHTML = imgTag; //adding that created img tag inside dropArea container
    }

    fileReader.readAsDataURL(file);
    // show generated caption, once the image is uploaded and shown
    // document.getElementById('caption').append('Lorem ipsum dolor sit amet consectetur adipisicing elit. Maxime esse asperiores tempora dicta commodi, nesciun')
    // show a button to remove the image 
    removeImageButton.style.display = "block";

  } else {
      alert("This is not an Image File!");
      dropArea.classList.remove("active");
      dragText.textContent = "Drag & Drop to Upload File";
  }
}

function rmvButtonClicked() {
  dropArea.innerHTML = da;
  document.getElementById('caption').innerText = '';
  removeImageButton.style.display = "none";
  file = null
}

function submitFormAndShowCaption(){
  //showing spinner
  spinner = document.getElementById("loader")
  spinner.style.display = "inline-block";
  //submitting form
  document.getElementById("form1").submit() 
  //stopping spinner

}