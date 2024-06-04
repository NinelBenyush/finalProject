import React from 'react';
import { Link as RouterLink } from 'react-router-dom';
import { links,profileLinks } from "../data";
import { FiMessageSquare } from "react-icons/fi";
import { MdAccountCircle } from "react-icons/md";
import { FaHome } from "react-icons/fa";

function ProfileNavbar(){


    const info = profileLinks.filter((link) => link.text!=="Messages")
    const message = profileLinks.filter((link) => link.text!=="Basic Information")
    const filteredLinks = links.filter((link) => link.text !== 'Log in');
    
    return (
        <nav className="bg-emerald-100">
            <div className="align-element py-4 flex flex-col sm:flex-row sm:gap-x-16 sm:items-center sm:py-8">
                <h2 className="text-3xl font-bold">
                    Order<span className="text-emerald-600">Boost</span>
                </h2>

                          
  <ul className="menu menu-horizontal flex gap-x-3">
  <li>
  {filteredLinks.map((link) => {
                        const { id, href, text} = link;
                        return (
                            <a key={id} href={href} className="tooltip" data-tip="Home">
                            <FaHome className="h-5 w-5" />
                            </a>
                        );
                    })}

  </li>
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

        </nav>

    )
}

export default ProfileNavbar;