import { useEffect } from 'react'
import { BrowserRouter, useLocation, useNavigate } from 'react-router-dom'
import { ToastContainer } from 'react-toastify'
import 'react-toastify/dist/ReactToastify.css'

import '../src/assets/css/base/root.css'
import Progress from './components/Progress'
import { useAppDispatch, useAppSelector } from './redux/hooks'
import { updateAuth } from './redux/thunk/authThunk'
import './reset_css.css'
import ApplicationRoute from './routes/ApplicationRoute'
import Loading from './utils/Loading'
import ScrollToTop from './utils/ScrollToTop'
import { Role } from './utils/roles'

function App() {
  const dispatch = useAppDispatch()
  const { loading } = useAppSelector((state) => state.auths)

  useEffect(() => {
    const fetchApiUpdateAuth = async () => {
      await dispatch(updateAuth())
    }
    fetchApiUpdateAuth()
  }, [])

  return (
    <>
      <BrowserRouter>
        <ScrollToTop />
        <>{loading ? <Loading /> : <ApplicationRoute />}</>{' '}
      </BrowserRouter>
      <ToastContainer
        position='top-right'
        autoClose={2000}
        hideProgressBar={false}
        newestOnTop={false}
        closeOnClick={false}
        rtl={false}
        pauseOnFocusLoss={false}
        draggable={false}
        pauseOnHover={false}
        theme='light'
      />
      <Progress />
    </>
  )
}

export default App
