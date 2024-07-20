import React from "react";
import { FiMessageSquare } from "react-icons/fi";
import { MdAccountCircle } from "react-icons/md";
import SectionTitle from "./SectionTitle";

function Profile() {
  
  return (
    <section className="bg-white py-20 ">
      <div className="align-element grid md:grid-cols-2 items-center gap-16">
<ul className="menu menu-horizontal bg-base-200 rounded-box mt-6">
  <li>
    <a className="tooltip" data-tip="Account">
    <MdAccountCircle className="h-5 w-5" />
    </a>
  </li>
  <li>
    <a className="tooltip" data-tip="Update Info">
      <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
    </a>
  </li>
  <li>
    <a className="tooltip" data-tip="Messages">
    <FiMessageSquare className="h-5 w-5" />
    </a>
  </li>
</ul>
 </div>
    </section>
  );
}

export default Profile;
