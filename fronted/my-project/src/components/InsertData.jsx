import img from "../assets/insertData2.jpg";

function InsertData(){
    return (
<div className="flex justify-center items-center min-h-screen">
  <div className="card card-side bg-base-100 shadow-xl" style={{ width: '600px' }}>
    <figure>
      <img src={img} alt="Movie"  />
    </figure>
    <div className="card-body">
      <h2 className="card-title">Here you need to choose a file</h2>
      <p>Click the button to upload</p>
      <div className="card-actions justify-end">
        <button className="btn bg-emerald-100">Upload</button>
      </div>
    </div>
  </div>
</div>


    )
}

export default InsertData;