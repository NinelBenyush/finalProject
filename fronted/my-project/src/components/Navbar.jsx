import React, { useState } from 'react';
import { Link as RouterLink } from 'react-router-dom';
import { links,profile, dropdownLinks } from "../data";

function Navbar() {
  const [dropdownOpen, setDropdownOpen] = useState(false);

  const toggleDropdown = () => {
    setDropdownOpen(!dropdownOpen);
  };

  return (
    <nav className="bg-emerald-100">
      <div className="align-element py-4 flex flex-col sm:flex-row sm:gap-x-16 sm:items-center sm:py-8">
        <h2 className="text-3xl font-bold">
          Order<span className="text-emerald-600">Boost</span>
        </h2>
        <div className="flex gap-x-3">
          {links.map((link) => {
            const { id, href, text } = link;
            return (
              <a key={id} href={href} className="capitalized text-lg tracking-wide hover:text-emerald-600 duration-300">
                {text}
              </a>
            );
          })}

           {profile.map((link) => {
            const { id, href, text } = link;
            return (
              <a key={id} href={href} className="capitalized text-lg tracking-wide hover:text-emerald-600 duration-300">
                {text}
              </a>
            );
          })}
          
          <div className="relative">
            <div 
              className="capitalized text-lg tracking-wide hover:text-emerald-600 duration-300 cursor-pointer"
              onClick={toggleDropdown}
            >
              More
            </div>
            {dropdownOpen && (
              <ul className="absolute z-10 bg-base-100 shadow rounded-box mt-2">
                {dropdownLinks.map((link) => {
                  const { id, href, text } = link;
                  return (
                    <li key={id}>
                      <a href={href} className="block px-4 py-2 hover:bg-emerald-600 hover:text-white">
                        {text}
                      </a>
                    </li>
                  );
                })}
              </ul>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
}

export default Navbar;
