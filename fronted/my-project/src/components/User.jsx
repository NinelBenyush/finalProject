function User({ username }) {
    const getChar = () => {
      if (username) {
        return username.charAt(0).toUpperCase();
      }
      return 'A';
    };
    console.log("User component, username:", username);
  
    return (
      <div className="avatar placeholder">
        <div className="bg-neutral text-neutral-content rounded-full w-8">
          <span className="text-xs">{getChar()}</span>
        </div>
      </div>
    );
  }
  
  export default User;
  